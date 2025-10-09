import cv2
import numpy as np

img = cv2.imread("img/images(1).jpeg")
img_copy = img.copy()
img = cv2.GaussianBlur(img, (3, 3), 5)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0,0,127]) #мінімальний поріг
upper = np.array([179,102,180]) #максимальний поріг
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img, img, mask = mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        perimeter = cv2.arcLength(cnt, True) #True - замкнутий контур
        M = cv2.moments(cnt) #центр мас - середня позиція контуру

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2) #допомагає відрізняти співвідношення сторін
        #міра округлості обєкта
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True) #точність апроксимації
        if len(approx) == 3: #якщо кфлькість вершин
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Square"
        elif len(approx) >8:
            shape = "Oval"
        else:
            shape = "Inshe"

        cv2.drawContours(img_copy, [cnt], -1, (0, 0, 255), 2)
        cv2.circle(img_copy, (cX, cY), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f"{shape}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img_copy, f"A:{int(area)}, P:{int(perimeter)}", (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img_copy, f"AR: {aspect_ratio}, C:{compactness}", (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

cv2.imshow("mask", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()