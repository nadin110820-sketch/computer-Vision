import cv2
import numpy as np

image = cv2.imread('image/img.png')
image =cv2.resize(image,(600,600))
cv2.rectangle(image,(280,100),(430,250),(255,255,255), 2)
cv2.putText(image, "Superskaya Nadia", (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()