import cv2
import ultralytics
from ultralytics import YOLO


model = YOLO('yolov8s-pose.pt')

video_path =0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    sucess,frame = cap.read()
    if sucess:
        results = model(frame, save=True)
        annoted = results[0].plot()

        cv2.imshow('out',annoted)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    else:
        break
cap.release()
cv2.destroyAllWindows()

