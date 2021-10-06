import numpy as np
import cv2
import imutils

im = cv2.imread('sample_images/base-long.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # chuyển ảnh xám thành ảnh grayscale

ret, thresh = cv2.threshold(im, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 2, 1)

for (index, contour) in enumerate(contours):
	(x, y, w, h) = cv2.boundingRect(contour)
	cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow("final", im)
cv2.waitKey(0)
cv2.destroyAllWindows()