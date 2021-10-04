from imutils import paths
import imutils
import cv2
import numpy as np

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



src='./sample_images/bien_ngang_nho.jpg'

image = cv2.imread(src)
image = imutils.resize(image, width=800)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)

sharpen = loop_sharp(gray,0)
# cv2.imshow('sharpen', sharpen)

rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
blackhat = cv2.morphologyEx(sharpen, cv2.MORPH_BLACKHAT, rectKern)
# cv2.imshow('blackhat', blackhat)

blackhat = loop_sharp(blackhat,0)

cv2.imshow('blackhat_2', blackhat)

squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)

light = cv2.threshold(light, 0, 255,
                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow('light', light)

# this will not work if the surrounding boundary is lost in transformation
# imagem = cv2.bitwise_not(light)
# cnts = cv2.findContours(imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# min_area = 100
# max_area = 1500
# for c in cnts:
# 	area = cv2.contourArea(c)
# 	if area > min_area and area < max_area:
# 			x,y,w,h = cv2.boundingRect(c)
# 			ROI = image[y:y+h, x:x+w]
# 			# cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
# 			cv2.rectangle(imagem, (x, y), (x + w, y + h), (36,255,12), 2)
# 			# image_number += 1

# cv2.imshow("light-drawn", imagem)

# This is not too reliable, as the text itself is really small
gradX = cv2.Sobel(
	blackhat,
	ddepth=cv2.CV_32F,
	dx=1, dy=0, ksize=-1
)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
# cv2.imshow("Scharr", gradX)

gradX = cv2.GaussianBlur(gradX, (3, 3), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
thresh = cv2.threshold(gradX, 0, 255,
cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow("Grad Thresh", thresh)

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
# cv2.imshow("Grad Erode/Dilate", thresh)


thresh = cv2.bitwise_and(thresh, thresh, mask=light)
# cv2.imshow("wise", thresh)


thresh = cv2.dilate(thresh, None, iterations=2)
thresh = cv2.erode(thresh, None, iterations=1)
cv2.imshow("Final", thresh)

ok_range = [(1.5, 2.5), (3, 4.5)]
def check_range(x,y,w,h):
	if h == 0:
		return False
	ratio = w/h
	for r in ok_range:
		if ratio>=r[0] and ratio <= r[1]:
			return True
	return False

# find plate ( rectangle-ish shape)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

min_area = 100
max_area = 1500
for c in cnts:
	area = cv2.contourArea(c)
	if area > min_area and area < max_area:
			x,y,w,h = cv2.boundingRect(c)
			ROI = image[y:y+h, x:x+w]
			ratio = w/h
			ok = check_range(x,y,w,h)
			print('ratio', ratio, (x,y), (h,w), ok)
			# cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
			if ok:
				cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
			else:
				pass
				# cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
			# image_number += 1

cv2.imshow("light-drawn", image)

cv2.waitKey(0)
cv2.destroyAllWindows()