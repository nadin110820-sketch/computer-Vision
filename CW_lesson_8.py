import cv2
import numpy as np

# face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2 (1).xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye (1).xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile (1).xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize = (30, 30)) #2 - масштабування 3 - кількість перевірок 4 - мінімальний розмір
    # print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 25, minSize = (15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 10, minSize = (20, 20))
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Tracking face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
