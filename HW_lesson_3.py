import cv2
import numpy as np
image = cv2.imread('images/img.png')

cv2.putText(image, 'Lynovitskaya Nadia', (490, 545), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
cv2.rectangle(image, (490, 240), (680, 520), (0, 0, 255), 2)
cv2.imshow('photo', image)
cv2.waitKey(0)
cv2.destroyAllWindows()