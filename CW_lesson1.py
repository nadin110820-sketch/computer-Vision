import cv2
import numpy as np

# image = cv2.imread('images/flower.jpg')
# # image =cv2.resize(image,(800,600))
# image =cv2.resize(image,(image.shape[1] // 4, image.shape[0] // 4))
# print(image.shape)
# # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# # image = cv2.flip(image, 1) #1 gorizontalno 0 verticalno
# # image = cv2.GaussianBlur(image,(5,5),7) #можна ставити тільки непарні
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image,100,250) #!
# # image = cv2.dilate(image,None,iterations = 3)
# kernel = np.ones((5,5),np.uint8)
# image = cv2.dilate(image,kernel,iterations = 3)
# image = cv2.erode(image,kernel,iterations = 1)
#
# print(image.shape)
# cv2.imshow('image', image)
# cv2.imshow('image', image[0:150, 0:100])

# video = cv2.VideoCapture('video/kangaroo.mp4')
video = cv2.VideoCapture(0)

while True:
    success, frame = video.read()
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.waitKey(0)
cv2.destroyAllWindows()