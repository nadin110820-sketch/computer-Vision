import cv2
import numpy as np

img = np.zeros((400,600,3), np.uint8)
img[:] = (47, 47, 146)

photo = cv2.imread("im/qr.png")
photo = cv2.resize(photo, (100, 100))
img[230:330, 450:550] = photo

qr = cv2.imread("im/img.png")
qr = cv2.resize(qr, (100, 100))
img[20:120, 20:120] = qr


cv2.rectangle(img,(10,10),(590,390),(0, 0, 0),4)

cv2.putText(img, "Lynovitskaya Nadia", (170, 100),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 1)
cv2.putText(img, "Computer Vision Student", (170, 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (51, 51, 255), 2)
cv2.putText(img, "Email: nadin110820@gmail.com", (170, 210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (64 , 64, 200), 1)
cv2.putText(img, "Phone: +380993457950", (170, 250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (64 , 64, 200), 1)
cv2.putText(img, "11.08.2010", (170, 290), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (64 , 64, 200), 1)
cv2.putText(img, "OpenCV Business Card", (170, 360), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (31, 33, 33), 2)


cv2.imshow("Image", img)
cv2.imwrite("business_card.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()