import cv2
import  numpy as np

img = np.zeros((512,512,3), np.uint8)

# img[:] = 142, 228, 126 #просто заливає все

# img[100:150,200:280] = 255 #заливає фрагмент

cv2.rectangle(img,(100,100),(200,200),(255,255,255), 2)
cv2.line(img,(100,100),(200,200),(255,255,255),2)
print(img.shape)
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (255, 255, 255))
cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (255, 255, 255), 1)

cv2.circle(img,(200,200), 20, (255,255,255), 1)
cv2.putText(img, "Bla bla", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()