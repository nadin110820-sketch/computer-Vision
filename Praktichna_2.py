import cv2
import numpy as np

img = cv2.imread("img/IMG.jpeg")
scale = 3
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
print(img.shape)

img_copy = img.copy()
img = cv2.GaussianBlur(img, (3, 3), 5)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = np.array([0, 76, 0])
upper_green = np.array([55, 255, 251])

lower_blue = np.array([106, 0, 132])
upper_blue = np.array([107, 243, 211])

lower_black = np.array([89, 14, 17])
upper_black = np.array([125, 73, 110])

mask_red = cv2.inRange(img, lower_black, upper_black)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_green = cv2.inRange(img, lower_green, upper_green)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)



contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Square"
        elif len(approx) >8:
            shape = "Oval"
        else:
            shape = "Inshe"

        x, y, w, h = cv2.boundingRect(cnt)
        roi = img[y:y + h, x:x + w]
        avg_color = cv2.mean(roi)[:3]

        h_value = avg_color[0]
        if 0 <= h_value <= 15 or 160 <= h_value <= 180:
            color_name = "Red"
        elif 35 <= h_value <= 85:
            color_name = "Green"
        elif 100 <= h_value <= 140:
            color_name = "Blue"
        else:
            color_name = "Other"


        cv2.drawContours(img_copy, [cnt], -1, (0, 0, 255), 1)
        cv2.circle(img_copy, (cX, cY), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f"{shape}", (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)
        cv2.putText(img_copy, f"A:{int(area)}, P:{int(perimeter)}", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(img_copy, f"AR: {aspect_ratio}, C:{compactness}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)
        cv2.putText(img_copy, color_name, (x - 35, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 0), 2)




# cv2.imshow("mask", img)
cv2.imwrite("Praktichna_2", img_copy)
cv2.imshow("mask", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()