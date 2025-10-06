import cv2
import numpy as np

img = cv2.imread('image/download(1).jpg')
scale = 2 #збільшення зменшення
img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale))
print(img.shape)

img_copy_color = img.copy()
img_copy = img.copy()


img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 4)
img_copy = cv2.equalizeHist(img_copy) #підсилення контрасту
img_copy = cv2.Canny(img_copy, 100, 110)

contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #2- знаходить лише крайні зовнішні контури і якщо лбєкт буде мати дірки, то будуть ігноруватися. 3 - ставимо ключові точки, апроксимація

#малювання - це контурінг
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 365: #фільтр шум #залежить від розміру фото
        x, y, w, h = cv2.boundingRect(cnt) #будуємо прямокутник
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y > 10 else y + 15
        text = f"x:{x}, y:{y}, S:{int(area)}"
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1 )

# cv2.imshow('image', img)
# cv2.imshow('img_copy', img_copy)
cv2.imshow('img_copy_color', img_copy_color)
cv2.waitKey(0)
cv2.destroyAllWindows()