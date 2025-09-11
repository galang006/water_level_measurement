from ultralytics import YOLO
import supervision as sv
import os
import cv2
import estimated_water_height_process as ewhp
import config

MODEL_WATER_GAUGE = config.MODEL_WATER_GAUGE
MODEL_NUMBER = config.MODEL_NUMBER
MODEL_WATER_SEGMENTATION = config.MODEL_WATER_SEGMENTATION

test_frame_loc = "dataset/test frames"
test_frame_list = ["malam 2", "malam 0", "malam 1", "malam cerah", "malam hujan" ,"siang 0", "siang 1", "siang 2", "siang cerah", "siang hujan"]
test_frame_list = ["siang 0"]

output_folder = "dataset/test frames/result"
os.makedirs(output_folder, exist_ok=True)

model_wg = YOLO(MODEL_WATER_GAUGE)
model_number = YOLO(MODEL_NUMBER)   
model_ws = YOLO(MODEL_WATER_SEGMENTATION)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()

for key, data in enumerate(test_frame_list):
    for i in range(1, 31):
        frame_name = f"{data}_frame_{i:02d}.jpg"
        frame_path = f"{test_frame_loc}/{frame_name}"

        print(f"Memproses frame: {frame_path}")
        if not os.path.exists(frame_path):
            print(f"Gagal membuka frame: {frame_path}")
            continue

        frame = cv2.imread(frame_path)
        frame_original = frame.copy()

        results_wg = model_wg(frame, conf=0.6)
        detections_wg = sv.Detections.from_ultralytics(results_wg[0])

        annotated_frame_wg = box_annotator.annotate(scene=frame, detections=detections_wg)
        annotated_frame_wg = label_annotator.annotate(scene=annotated_frame_wg, detections=detections_wg)

        if len(detections_wg.xyxy) > 0:
            x1, y1, x2, y2 = detections_wg.xyxy[0].astype(int)
            roi = frame[y1:y2, x1:x2]

            results_number = model_number(roi, conf=0.6)
            detections_number = sv.Detections.from_ultralytics(results_number[0])

            annotated_frame_number = box_annotator.annotate(scene=roi, detections=detections_number)
            annotated_frame_number = label_annotator.annotate(scene=annotated_frame_number, detections=detections_number)

            result_water_segmentation = model_ws(roi, conf=0.6)
            detections_ws = sv.Detections.from_ultralytics(result_water_segmentation[0])
            annotated_frame_ws = mask_annotator.annotate(scene=roi, detections=detections_ws)

            if len(detections_ws.xyxy) > 0:
                x1_seg, y1_seg, x2_seg, y2_seg = map(int, detections_ws.xyxy[0])

                x_center = (x1_seg + x2_seg) // 2
                y_center = (y1_seg + y2_seg) // 2

                water_height = ewhp.calculate_estimated_water_height(detections_wg, detections_number, detections_ws)

                cv2.putText(annotated_frame_ws, str(water_height), (x_center-20, y_center), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
            annotated_seg_resized = cv2.resize(annotated_frame_ws, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            annotated_frame_wg[y1:y2, x1:x2] = annotated_seg_resized

        output_path = os.path.join(output_folder, f"{data}_result_frame_{i:02d}.jpg")
        cv2.imwrite(output_path, annotated_frame_wg)
        
