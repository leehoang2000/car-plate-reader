import numpy as np
import cv2
import imutils

window_name = 'Image'

# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 0.5

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 1


def create_blank(width, height, rgb_color=(0, 0, 0)):
	"""Create new image(numpy array) filled with certain color in RGB"""
	# Create black blank image
	image = np.zeros((height, width, 3), np.uint8)
	
	# Since OpenCV uses BGR, convert the color first
	color = tuple(reversed(rgb_color))
	# Fill image with color
	image[:] = color
	
	return image


im = cv2.imread('sample_images/base-clean.jpg')

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # chuyển ảnh xám thành ảnh grayscale

imgray = cv2.Canny(imgray, 127, 255)  # nhị phân hóa ảnh

contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for (index, contour) in enumerate(contours):
	(x, y, w, h) = cv2.boundingRect(contour)
	ratio = w / float(h)
	if ((ratio > 1.2 and ratio < 2) or (ratio > 3 and ratio < 5.5)) and index == 432:
		# approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
		
		# cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
		im[:,:] = (255,255,255)
		im[x : x+h, y: y + w] = (0, 0, 0)
		# cv2.putText(im, str(index), (x, y), font, fontScale, color, thickness, cv2.LINE_AA)

# cv2.drawContours(im, contours, -1, (0, 255, 0), 2)  # vẽ lại ảnh contour vào ảnh gốc

# show ảnh lên
cv2.imshow("final", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('base-long.jpg', im)
