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


def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
	"""Return a sharpened version of the image, using an unsharp mask."""
	blurred = cv2.GaussianBlur(image, kernel_size, sigma)
	sharpened = float(amount + 1) * image - float(amount) * blurred
	sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
	sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
	sharpened = sharpened.round().astype(np.uint8)
	if threshold > 0:
		low_contrast_mask = np.absolute(image - blurred) < threshold
		np.copyto(sharpened, image, where=low_contrast_mask)
	return sharpened


def loop_sharp(img, iteration):
	for _ in range(iteration):
		img = unsharp_mask(img)
	return img


im = cv2.imread('sample_images/bien_do.jpg')
im = loop_sharp(im, 1)
img = imutils.resize(im, width=900)

image = img.copy()
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # chuyển ảnh xám thành ảnh grayscale

blur = cv2.GaussianBlur(imgray, (5, 5), 0)
thresh = blur
cv2.imshow("thresh", thresh)

squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
light = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, squareKern)
light = cv2.GaussianBlur(light, (5, 5), 0)
light = cv2.erode(light, None, iterations=2)
light = cv2.dilate(light, None, iterations=2)
light = cv2.morphologyEx(light, cv2.MORPH_OPEN, squareKern)
# light = cv2.threshold(light, 0, 255,
#                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("lightq", light)
#
# alpha = 1.5  # Contrast control (1.0-3.0)
# beta = 0  # Brightness control (0-100)
#
# adjusted = cv2.convertScaleAbs(light, alpha=alpha, beta=beta)
# cv2.imshow("adjusted", adjusted)


canny = cv2.Canny(light, 127, 255)  # nhị phân hóa ảnh
# cv2.imshow("canny", canny)

#
# thresh = imgray
# cv2.imshow("thresh", thresh)
#
#
# sharpen = loop_sharp(thresh, iteration=1)
#
# rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
# blackhat = cv2.morphologyEx(sharpen, cv2.MORPH_BLACKHAT, rectKern)
# cv2.imshow("blackhat", blackhat)
#
# blackhat = cv2.dilate(blackhat, None, iterations=1)
# blackhat = cv2.erode(blackhat, None, iterations=1)
#
# cv2.imshow("blackhat-fill", blackhat)
#
# imagem = cv2.bitwise_not(blackhat)

imgray = loop_sharp(imgray, 1)
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
blackhat = cv2.morphologyEx(imgray, cv2.MORPH_BLACKHAT, rectKern)
cv2.imshow("blackhat", blackhat)

gradX = cv2.Sobel(
	blackhat,
	ddepth=cv2.CV_32F,
	dx=1, dy=0, ksize=-1
)
# gradX = cv2.erode(gradX, None, iterations=2)
# gradX = cv2.dilate(gradX, None, iterations=2)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
cv2.imshow("Scharr", gradX)

gradX = cv2.GaussianBlur(gradX, (3, 3), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)

thresh = cv2.threshold(gradX, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.dilate(thresh, None, iterations=1)
thresh = cv2.erode(thresh, None, iterations=1)
cv2.imshow("Grad Erode/Dilate", thresh)

sm_mask = cv2.bitwise_or(light, thresh)
# thresh = cv2.dilate(thresh, None, iterations=1)
# thresh = cv2.erode(thresh, None, iterations=1)
cv2.imshow("mask", sm_mask)
cannyt = cv2.Canny(sm_mask, 127, 255)  # nhị phân hóa ảnh
cv2.imshow("canny_", cannyt)

#
end = loop_sharp(cannyt, 0)

contours, _ = cv2.findContours(end, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
	(x, y, w, h) = cv2.boundingRect(contour)
	ratio = w / float(h)
	if (w > 5 and h > 10 and (ratio > 1.2 and ratio < 2) or (ratio > 3 and ratio < 5.5)):
		approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(img, str(len(approx)), (x, y), font, fontScale, color, thickness, cv2.LINE_AA)

# cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # vẽ lại ảnh contour vào ảnh gốc

# show ảnh lên
cv2.imshow("final", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
