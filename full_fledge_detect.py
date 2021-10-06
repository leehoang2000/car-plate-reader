# import pytesseract
import numpy as np
import imutils
import cv2


# from google.colab.patches import cv2_imshow

def cv2_imshow(image):
	cv2.imshow('hello', image)
	cv2.waitKey(0)


class FullFledgeDetect:
	def __init__(self, debug=True):
		self.debug = debug
		self.minAR = 1.2
		self.maxAR = 6
		self.keep = 20
	
	def debug_imshow(self, title, image, waitKey=False):
		# colab has different image display style
		if self.debug:
			print(title)
			cv2_imshow(image)
	
	def unsharp_mask(self, image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
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
	
	def loop_sharp(self, img, iteration):
		for _ in range(iteration):
			img = self.unsharp_mask(img)
		return img
	
	def get_candidates(self, gray, keep=5):
		
		sharpen = self.loop_sharp(gray, 3)
		self.debug_imshow('sharpen', sharpen)
		
		rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
		blackhat = cv2.morphologyEx(sharpen, cv2.MORPH_BLACKHAT, rectKern)
		self.debug_imshow('blackhat', blackhat)
		
		blackhat = self.loop_sharp(blackhat, 0)
		# add_show('blackhat_2', blackhat)
		
		squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
		light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
		
		# blurred = cv2.GaussianBlur(gray, (3, 3), 0)
		
		wide = cv2.Canny(blackhat, 10, 200)
		wide = cv2.dilate(wide, None, iterations=2)
		wide = cv2.erode(wide, None, iterations=1)
		
		self.debug_imshow('wide', wide)
		
		# mid = cv2.Canny(blackhat, 30, 150)
		# self.debug_imshow('mid', mid)
		
		# tight = cv2.Canny(blackhat, 240, 250)
		# self.debug_imshow('tight', tight)
		
		light = cv2.threshold(light, 0, 255,
		                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		self.debug_imshow('light', light)
		
		sm_mask = cv2.bitwise_or(light, wide)
		self.debug_imshow('mk', sm_mask)
		
		# this will not work if the surrounding boundary is lost in transformation
		# imagem = cv2.bitwise_not(light)
		# cnts = cv2.findContours(imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		# self.debug_imshow("light-drawn", imagem)
		
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
		self.debug_imshow("Scharr", gradX)
		
		gradX = cv2.GaussianBlur(gradX, (3, 3), 0)
		gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
		thresh = cv2.threshold(gradX, 0, 255,
		                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		self.debug_imshow("Grad Thresh", thresh)
		
		thresh = cv2.dilate(thresh, None, iterations=1)
		thresh = cv2.erode(thresh, None, iterations=1)
		self.debug_imshow("Grad Erode/Dilate", thresh)
		
		thresh = cv2.bitwise_and(thresh, sm_mask)
		# self.debug_imshow("wise", thresh)
		
		thresh = cv2.dilate(thresh, None, iterations=1)
		thresh = cv2.erode(thresh, None, iterations=1)
		self.debug_imshow("Final", thresh)
		
		# thresh = cv2.bitwise_and(tight, thresh)
		# self.debug_imshow("vel", thresh)
		
		cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:self.keep]
		
		return cnts
	
	def locate_license_plate(self, gray, candidates):
		if candidates is None:
			return (None, None)
		
		lpCnt = None
		roi = None
		for c in candidates:
			(x, y, w, h) = cv2.boundingRect(c)
			ar = w / float(h)
			
			if ar >= self.minAR and ar <= self.maxAR:
				# store the license plate contour and extract the
				# license plate from the grayscale image and then
				# threshold it
				lpCnt = c
				licensePlate = gray[y:y + h, x:x + w]
				roi = cv2.threshold(licensePlate, 0, 255,
				                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
				# display any debugging information and then break
				# from the loop early since we have found the license
				# plate region
				# maybe disable this for multiple plates
				self.debug_imshow("License Plate", licensePlate)
				self.debug_imshow("ROI", roi, waitKey=True)
				break
		return (roi, lpCnt)
	
	def build_tesseract_options(self, psm=7):
		# tell Tesseract to only OCR alphanumeric characters
		alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
		options = "-c tessedit_char_whitelist={}".format(alphanumeric)
		# set the PSM mode
		options += " --psm {}".format(psm)
		# return the built options string
		return options
	
	# image loaded with cv2.imread
	def find_and_ocr(self, image, raw=False):
		lpText = None
		
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		self.debug_imshow('gray scale', gray)
		
		candidates = self.get_candidates(gray)
		if raw is True:
			return candidates
		
		(lp, lpCnt) = self.locate_license_plate(gray, candidates)
		
		if lp is not None:
			options = self.build_tesseract_options()
			lpText = pytesseract.image_to_string(lp, config=options)
			self.debug_imshow("License Plate", lp)
		
		return (lpText, lpCnt)


if __name__ == '__main__':
	anpr = FullFledgeDetect(debug=True)
	imagePaths = ['./sample_images/bien_do.jpg']
	
	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		
		candidates = anpr.find_and_ocr(image, raw=True)
		
		for c in candidates:
			area = cv2.contourArea(c)
			x, y, w, h = cv2.boundingRect(c)
			ROI = image[y:y + h, x:x + w]
			ratio = w / h

			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
			# cv2.rectangle(thresh, (x, y), (x + w, y + h), (255,0,0), 2)
		cv2_imshow(image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# only continue if the license plate was successfully OCR'd
	# if lpText is not None and lpCnt is not None:
	# 	box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
	# 	box = box.astype("int")
	# 	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	
	# 	(x, y, w, h) = cv2.boundingRect(lpCnt)
	# 	cv2_imshow(image)
