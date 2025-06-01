import streamlit as st
import tempfile
import estimated_water_height_process as ewhp
import os
import time
from dotenv import load_dotenv
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
import supervision as sv

load_dotenv()

WATER_GAUGE_THRESHOLD = float(os.getenv("WATER_GAUGE_THRESHOLD"))
NUMBER_OBJECT_DETECTION_THRESHOLD = float(os.getenv("NUMBER_OBJECT_DETECTION_THRESHOLD"))
WATER_SEGMENTATION_THRESHOLD = float(os.getenv("WATER_SEGMENTATION_THRESHOLD"))
MODEL_WATER_GAUGE = os.getenv("MODEL_WATER_GAUGE")
MODEL_NUMBER = os.getenv("MODEL_NUMBER")
MODEL_WATER_SEGMENTATION = os.getenv("MODEL_WATER_SEGMENTATION")

def draw_supervision(image, result, label=""):
    detections = sv.Detections.from_ultralytics(result[0])
    box_annotator = sv.BoxAnnotator()
    labels = [f"{label} {r:.2f}" for r in detections.confidence]
    annotated = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    return annotated

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name
    output_path = f"hasil/{video_file.name}"

    st.video(video_file) 

    st.subheader("Set Confident Threshold Parameters")

    wg_threshold = st.slider("WATER GAUGE Threshold", 0.0, 1.0, WATER_GAUGE_THRESHOLD)
    n_threshold = st.slider("NUMBER OBJECT DETECTION Threshold", 0.0, 1.0, NUMBER_OBJECT_DETECTION_THRESHOLD)
    ws_threshold = st.slider("WATER SEGMENTATION Threshold", 0.0, 1.0, WATER_SEGMENTATION_THRESHOLD)

    if st.button("Process Video"):
        progress_bar = st.progress(0)
    
        def update_progress(p):
            progress_bar.progress(min(int(p * 100), 100))

        with st.spinner("Processing video... Please wait."):
            start_time = time.time()
            avg_water_height = ewhp.detect(
                video_path, output_path,
                wg_threshold, n_threshold, ws_threshold,
                progress_callback=update_progress
            )
            end_time = time.time()

        st.success(f"Processing complete in {end_time - start_time:.2f} seconds!")
        st.video(output_path)
        st.success(f"Average Water Height : {avg_water_height} cm")

    tfile.close()

def process_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    st.image(img_rgb,)

    select_model = st.radio("Select Model", ("All", "Water Gauge", "Number Object Detection", "Water Segmentation"))

    if select_model == "All":
        process_image_using_all_models(img)
    elif select_model == "Water Gauge":
        model_path = MODEL_WATER_GAUGE
    elif select_model == "Number Object Detection":
        model_path = MODEL_NUMBER
    elif select_model == "Water Segmentation":
        model_path = MODEL_WATER_SEGMENTATION
    else:
        st.error("Seelct a valid model")

    if select_model != "All":
        st.subheader("Set Confidence Threshold")
        wg_threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
        if st.button("Process Image", key=1):
            model = YOLO(model_path)
            results = model.predict(source=img, conf=wg_threshold)
            annotated_img = results[0].plot(font_size=10, line_width=1)
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img_rgb, caption="Detection Result")

def process_image_using_all_models(image_file):
    st.subheader("Set Confidence Threshold")
    wg_threshold = st.slider("WATER GAUGE Threshold", 0.0, 1.0, WATER_GAUGE_THRESHOLD)
    n_threshold = st.slider("NUMBER OBJECT DETECTION Threshold", 0.0, 1.0, NUMBER_OBJECT_DETECTION_THRESHOLD)
    ws_threshold = st.slider("WATER SEGMENTATION Threshold", 0.0, 1.0, WATER_SEGMENTATION_THRESHOLD)

    if st.button("Process Image", key=2):
        model_water_gauge = YOLO(MODEL_WATER_GAUGE)
        model_number = YOLO(MODEL_NUMBER)
        model_water_seg = YOLO(MODEL_WATER_SEGMENTATION)

        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=1)
        mask_annotator = sv.MaskAnnotator()

        results_wg = model_water_gauge(image_file, conf=wg_threshold)
        detections_wg = sv.Detections.from_ultralytics(results_wg[0])

        image_wg = image_file.copy()
        image_wg = box_annotator.annotate(scene=image_wg, detections=detections_wg)
        image_wg = label_annotator.annotate(scene=image_wg, detections=detections_wg)

        st.image(cv2.cvtColor(image_wg, cv2.COLOR_BGR2RGB), caption="Water Gauge Detection Result")

        if len(detections_wg.xyxy) > 0:
            x1, y1, x2, y2 = detections_wg.xyxy[0].astype(int)
            roi = image_file[y1:y2, x1:x2]

            results_number = model_number(roi, conf=n_threshold)
            detections_number = sv.Detections.from_ultralytics(results_number[0])

            roi_annotated = roi.copy()
            roi_annotated = box_annotator.annotate(scene=roi_annotated, detections=detections_number)
            roi_annotated = label_annotator.annotate(scene=roi_annotated, detections=detections_number)

            image_number_overlayed = image_wg.copy()
            image_number_overlayed[y1:y2, x1:x2] = roi_annotated

            st.image(cv2.cvtColor(image_number_overlayed, cv2.COLOR_BGR2RGB), caption="Number Object Detection Result")

            results_seg = model_water_seg(roi_annotated, conf=ws_threshold)
            detections_seg = sv.Detections.from_ultralytics(results_seg[0])

            roi_seg = roi_annotated.copy()
            roi_seg = mask_annotator.annotate(scene=roi_seg, detections=detections_seg)

            if len(detections_seg.xyxy) > 0:
                x1_seg, y1_seg, x2_seg, y2_seg = map(int, detections_seg.xyxy[0])
                
                x_center = (x1_seg + x2_seg) // 2
                y_center = (y1_seg + y2_seg) // 2

                water_height = str(ewhp.calculate_estimated_water_height(detections_wg, detections_number, detections_seg))

                if isinstance(water_height, (int, float)):  
                    sum_water_height += water_height
                    count_valid_frames += 1

                cv2.putText(roi_seg, str(water_height), (x_center-20, y_center), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
            
            image_final = image_wg.copy()
            image_final[y1:y2, x1:x2] = roi_seg

            st.image(cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB), caption="Water Segmentation Result")

            st.success(f"Estimate Water Height : {water_height} cm")

        else:
            st.warning("No water gauge detected. Cannot proceed to number and segmentation detection.")

if __name__ == '__main__':
    st.title("Estimated Water Height Detection")
    input_file = st.file_uploader("Upload a video or image", type=["mp4", "jpg", "jpeg", "png"])

    if input_file is not None:
        file_ext = Path(input_file.name).suffix.lower()

        if file_ext == ".mp4":
            process_video(input_file)
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            process_image(input_file)
        else:
            st.error("Unsupported file type")
            