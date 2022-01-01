import cv2
import os
os.environ['DISPLAY'] = ':0'
img = cv2.imread('test_enhance.png')
cv2.imshow("After enhance", img)
cv2.waitKey(0)
print(img)