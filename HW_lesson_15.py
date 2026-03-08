import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = False
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, 'catsdogs.mp4')
    cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO('yolov8s.pt')

CONF_THRESHOLD = 0.4
RESIZE_WIDTH = 960

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

prev_time = time.time()
FPS = 0.0

total_cats = 0
total_dogs = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    result = model(frame, conf=CONF_THRESHOLD, verbose=False)
    frame_cats = 0
    frame_dogs = 0
    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == CAT_CLASS_ID:
                frame_cats += 1
                total_cats += 1
                color = (255, 0, 255)
                label = f'Cat {conf:.2f}'
            elif cls == DOG_CLASS_ID:
                frame_dogs += 1
                total_dogs += 1
                color = (0, 255, 255)
                label = f'Dog {conf:.2f}'
            else:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, color, 2)
    total_animals = total_cats + total_dogs
    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        FPS = 1.0 / dt
    cv2.putText(frame, f'Cats: {frame_cats}', (20, 40),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    cv2.putText(frame, f'Dogs: {frame_dogs}', (20, 80),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Total animals: {total_animals}', (20, 120),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS: {FPS:.1f}', (20, 160),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.imshow('YOLO', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()