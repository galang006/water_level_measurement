import os
import cv2
from ultralytics import YOLO
import supervision as sv
from dotenv import load_dotenv
import subprocess

load_dotenv()

VIDEO_PATH = os.getenv("VIDEO_PATH")
OUTPUT_VIDEO_PATH = os.getenv("OUTPUT_VIDEO_PATH")
WATER_GAUGE_THRESHOLD = float(os.getenv("WATER_GAUGE_THRESHOLD"))
NUMBER_OBJECT_DETECTION_THRESHOLD = float(os.getenv("NUMBER_OBJECT_DETECTION_THRESHOLD"))
WATER_SEGMENTATION_THRESHOLD = float(os.getenv("WATER_SEGMENTATION_THRESHOLD"))
WATER_GAUGE_HEIGHT = int(os.getenv("WATER_GAUGE_HEIGHT"))

MODEL_WATER_GAUGE = os.getenv("MODEL_WATER_GAUGE")
MODEL_NUMBER = os.getenv("MODEL_NUMBER")
MODEL_WATER_SEGMENTATION = os.getenv("MODEL_WATER_SEGMENTATION")

def get_y_coor_number(detection):
    '''
        Mendapatkan koordinat y dari objek angka yang terdeteksi pada gambar.
        Returns:
            dict_center (dict): Dictionary yang berisi class_name / angka sebagai key dan koordinat y sebagai value serta diurutkan berdasarkan class_name.
    '''
    dict_center = {}
    for i in range(len(detection)):
        x1, y1, x2, y2 = map(int, detection.xyxy[i])
        center_y = (y1 + y2) // 2
        class_name = int(detection.data['class_name'][i])
        dict_center[class_name] = center_y

    sorted_data = dict(sorted(dict_center.items(), key=lambda item: int(item[0])))
    return sorted_data

def get_number_bounding_box_height(detection):
    '''
        Mendapatkan tinggi bounding box dari objek angka yang terdeteksi pada gambar.
        Returns:
            bounding_box_height (int): Tinggi bounding box dari objek angka yang terdeteksi.
    '''
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    #print(detection[1])
    return abs(y2 - y1)

def get_roi_length(detection):
    '''
        Mendapatkan panjang ROI dari objek water gauge yang terdeteksi pada gambar.
        Returns:
            roi_length (int): Panjang ROI dari objek water gauge yang terdeteksi.
    '''
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    return abs(x2 - x1)

def get_water_height(detection):
    '''
        Mendapatkan tinggi air dari objek water segmentation yang terdeteksi pada gambar.
        Returns:
            water_height (int): Tinggi air dari objek water segmentation yang terdeteksi.
    '''
    x1, y1, x2, y2 = map(int, detection.xyxy[0])
    return y1

def get_pixel_scale(detection, number_detections):
    '''
        Menghitung skala piksel berdasarkan objek angka yang terdeteksi pada gambar.
        Returns:
            pixel_scale (float): Skala piksel berdasarkan objek angka yang terdeteksi.
    '''
    if number_detections > 1:
        coor_y_number = get_y_coor_number(detection) # detection = detection_number

        keys = list(coor_y_number.keys())
        values = list(coor_y_number.values())

        pixel_scale = abs((keys[1] - keys[0]) / (values[1] - values[0]))
        return pixel_scale
        
    elif number_detections == 1:
        bounding_box_height = get_number_bounding_box_height(detection) # detection = detection_number
        pixel_scale = 20 / bounding_box_height
        return pixel_scale
    
    else:
        roi_length = get_roi_length(detection) # detection = detection_water_gauge
        pixel_scale = 120 / roi_length
        return pixel_scale

def calculate_estimated_water_height(detection_water_gauge, detection_number, detection_water_seg):
    '''
        Menghitung estimasi tinggi air berdasarkan deteksi objek water gauge, angka, dan segmentasi air.
        Returns:
            estimated_water_height (int): Estimasi tinggi air berdasarkan deteksi objek water gauge, angka, dan segmentasi air.
    '''
    number_detections = len(detection_number.xyxy)
    water_height = get_water_height(detection_water_seg)
    number_coordinates = get_y_coor_number(detection_number)

    if number_detections > 1:
        pixel_scale = get_pixel_scale(detection_number, number_detections)
        lower_number = next(iter(number_coordinates))
        coor_lower_number = number_coordinates[lower_number]
        water_distance = water_height - coor_lower_number

        water_distance_cm = water_distance * pixel_scale

        estimated_water_height = round(lower_number - water_distance_cm)
        # if estimated_water_height == 116:
            # print(f"number_coordinates: {number_coordinates}")
            # print(f"skala pixel : {pixel_scale}")
            # print(f"tinggi air : {water_height}")
            # print(f"jarak air : {water_distance}")
            # print(f"jarak air cm : {water_distance_cm}")
            # print(f"estimated_water_height : {estimated_water_height}")
            # print(f"bounding_box_height : {get_number_bounding_box_height(detection_number)}")
            # print(f"roi_length : {get_roi_length(detection_water_gauge)}\n")

        if estimated_water_height < 0:
            return 0
        elif estimated_water_height > WATER_GAUGE_HEIGHT:
            return WATER_GAUGE_HEIGHT
        else:
            return estimated_water_height


    elif number_detections == 1:
        pixel_scale = get_pixel_scale(detection_number, number_detections)
        number = next(iter(number_coordinates))
        coor_number = number_coordinates[number]
        water_distance = water_height - coor_number

        water_distance_cm = water_distance * pixel_scale

        estimated_water_height = round(number - water_distance_cm)
        if estimated_water_height < 0:
            return 0
        elif estimated_water_height > WATER_GAUGE_HEIGHT:
            return WATER_GAUGE_HEIGHT
        else:
            return estimated_water_height

    elif number_detections == 0:
        pixel_scale = get_pixel_scale(detection_water_gauge, number_detections)
     
        water_distance_cm = water_height * pixel_scale

        estimated_water_height = round(WATER_GAUGE_HEIGHT - water_distance_cm)
        if estimated_water_height < 0:
            return 0
        elif estimated_water_height > WATER_GAUGE_HEIGHT:
            return WATER_GAUGE_HEIGHT
        else:
            return estimated_water_height

def detect(video_path=VIDEO_PATH, output_path=OUTPUT_VIDEO_PATH, water_gauge_threshold=WATER_GAUGE_THRESHOLD,
            number_object_detection_threshold=NUMBER_OBJECT_DETECTION_THRESHOLD, water_segmentation_threshold=WATER_SEGMENTATION_THRESHOLD):
    '''
        Fungsi utama untuk mendeteksi objek pada video dan menghitung estimasi tinggi air.
        Menggunakan model YOLOv8n dan YOLOv8n-seg untuk mendeteksi objek water gauge, angka, dan segmentasi air.
    '''
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_output_path = output_path.replace(".mp4", "_raw.mp4")  # Sementara
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    model_water_gauge = YOLO(MODEL_WATER_GAUGE)          
    model_number = YOLO(MODEL_NUMBER) 
    model_water_seg = YOLO(MODEL_WATER_SEGMENTATION)   
    
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_original = frame.copy()
        frame_height, frame_width, _ = frame.shape

        results = model_water_gauge(frame, conf=water_gauge_threshold, verbose=False)[0]
        detections_water_gauge = sv.Detections.from_ultralytics(results)

        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections_water_gauge)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections_water_gauge)

        if len(detections_water_gauge.xyxy) > 0:
            x1, y1, x2, y2 = map(int, detections_water_gauge.xyxy[0])
            roi = frame_original[y1:y2, x1:x2]

            results_number = model_number(roi, conf=number_object_detection_threshold, verbose=False)[0]
            detections_number = sv.Detections.from_ultralytics(results_number)

            annotated_image_number = bounding_box_annotator.annotate(scene=roi, detections=detections_number)
            annotated_image_number = label_annotator.annotate(scene=annotated_image_number, detections=detections_number)

            results_water_seg = model_water_seg(roi, conf=water_segmentation_threshold, verbose=False)[0]
            detections_water_seg = sv.Detections.from_ultralytics(results_water_seg)

            annotated_image_water_seg = mask_annotator.annotate(scene=roi, detections=detections_water_seg)
            
            if len(detections_water_seg.xyxy) > 0:
                x1_seg, y1_seg, x2_seg, y2_seg = map(int, detections_water_seg.xyxy[0])
                
                x_center = (x1_seg + x2_seg) // 2
                y_center = (y1_seg + y2_seg) // 2

                text = str(calculate_estimated_water_height(detections_water_gauge, detections_number, detections_water_seg))

                cv2.putText(annotated_image_water_seg, text, (x_center-20, y_center), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
                
            annotated_image_number_resized = cv2.resize(annotated_image_water_seg, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            annotated_image[y1:y2, x1:x2] = annotated_image_number_resized

        cv2.namedWindow("Deteksi Objek", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Deteksi Objek", 1980, 1080)
        cv2.imshow('Deteksi Objek', annotated_image)

        out.write(annotated_image)
        
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", temp_output_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path  # overwrite hasil
    ]
    subprocess.run(ffmpeg_cmd)

    os.remove(temp_output_path)
    
if __name__ == '__main__':
    detect()
