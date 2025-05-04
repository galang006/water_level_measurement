
from ultralytics import YOLO
import cv2

# Load a COCO-pretrained YOLOv8n model
model = YOLO("models/water_gauge.pt")

# Display model information (optional)
model.info()

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("dataset/water_gauge/m_3_frame06.jpg", show=True)

cv2.imshow("Image", results[0].plot())
