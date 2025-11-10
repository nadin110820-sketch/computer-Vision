import cv2
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 128, 255),
    "purple": (255, 0, 255),
    "pink": (180, 105, 255),
    "white": (255, 255, 255)
}

X = []
y = []
noise_range = 20
samples_per_color = 100

for name, bgr in colors.items():
    base = np.array(bgr, dtype=np.int16)
    for _ in range(samples_per_color):
        noise = np.random.randint(-noise_range, noise_range + 1, size=3)
        sample = np.clip(base + noise, 0, 255)
        X.append(sample)
        y.append(name)

X = np.array(X, dtype=np.float32)
y = np.array(y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Не вдалося відкрити камеру")
    exit()

kernel = np.ones((5, 5), np.uint8)
saved_frames = 0
target_color = "red"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_counts = Counter()
    found_target_color = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue

        x, y_box, w, h = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        corners = len(approx)
        aspect = w / float(h) if h != 0 else 0

        if corners == 3:
            shape = "triangle"
        elif corners == 4:
            if 0.9 <= aspect <= 1.1:
                shape = "square"
            else:
                shape = "rectangle"
        elif corners >= 6:
            shape = "circle"
        else:
            shape = "unknown"

        roi = frame[y_box:y_box + h, x:x + w]
        if roi.size == 0:
            continue

        mean_color = cv2.mean(roi)[:3]
        mean_color = np.array(mean_color, dtype=np.float32).reshape(1, -1)
        color_label = model.predict(mean_color)[0]

        color_counts[color_label] += 1
        if color_label == target_color:
            found_target_color = True

        cv2.rectangle(frame, (x, y_box), (x + w, y_box + h), (255, 255, 255), 2)
        cv2.putText(frame, f"{color_label} {shape}", (x, y_box - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if color_counts:
        summary = ", ".join(f"{cnt} {clr}" for clr, cnt in color_counts.items())
        cv2.putText(frame, summary, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if found_target_color:
        filename = f"frame_{target_color}_{saved_frames}.png"
        cv2.imwrite(filename, frame)
        saved_frames += 1

    cv2.imshow("Color & shape detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
