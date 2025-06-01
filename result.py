from ultralytics import YOLO
import cv2

model = YOLO("models/number.pt")

frame = cv2.imread("dataset/number/clean/m_01_frame_05.jpg")
results = model(frame, save = False)
img = results[0].plot(font_size=1, line_width=1)

cv2.imshow("Image",img)

cv2.waitKey(0)
cv2.destroyAllWindows()