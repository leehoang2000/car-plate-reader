from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

class PyImageSearchANPR:
    def __init__(self, minAR=4, maxAR=5, debug=False):
        self.minAR = minAR
        self.maxAR = maxAR
        self.debug = debug

    def debug_imshow(self, title, image, waitKey=False):
        if self.debug:
            cv2.imshow(title,image)

            if waitKey:
                cv2.waitKey(0)
    
    def locate_licence_plate_candidates(self, gray, keep=5):
        #Blackhat the image to reveal black characters on white background
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_imshow("Blackhat", blackhat)

        #Fill small holes and reveal large light regions
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light,0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions", light)

        # Compute Scharr gradient magnitude in x-direction of blackhat
        # scale it back to range [0,255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255*((gradX - minVal)/(maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr",gradX)

        # smooth to group regions that may contain boundaries
        # Apply closing operation (to fill holes) and another binary threshold
        gradX = cv2.GaussianBlur(gradX, (5,5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh",thresh)

        # series of erosion and dilations to clean up the thresholded image
        thresh = cv2.erode(thresh, None, iterations = 2)
        thresh = cv2.dilate(thresh, None, iterations = 2)
        self.debug_imshow("Grad Erode/Dilate", thresh)

        # take the bitwise AND between threshold result and the light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final",thresh, waitKey=True)

        # Find the contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        # Return the list of contours
        return cnts
    
    def locate_license_plate(self, gray, candidates, clearBorder=False):
        lpCnt = None
        roi = None

        for c in candidates:
            # compute the bounding box of the contour and
            # then use the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w/ float(h)

            # check to see if the aspect ratio is rectangular
            if ar >= self.minAR and ar <= self.maxAR:
                # store the license plate contour and extract the 
                # license plate from the grayscale image and then 
                # threshold it
                lpCnt = c
                licensePlate = gray[y:y+h, x:x+h]
                roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                
                # check to see if we should clear any foreground
                # pixels touching the border of the image
                # (which typically, but not always, indicates noise)
                if clearBorder:
                    roi = clear_border(roi)

                # display any debugging information and then break
                # from the top early since we have found the license
                # plate region
                self.debug_imshow("License Plate", licensePlate)
                self.debug_imshow("ROI", roi, waitKey=True)
                break
        
        return(roi, lpCnt)
    
    #Page Segmentation Method (PSM): psm=7 "treat the image as a single text line"
    def build_tesseract_options(self, psm=6):
        # tell Tesseract to only OCR alphanumerical characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)

        # set the PSM mode
        options += " --psm {}".format(psm)

        # return the built options string
        return options
    
    def find_and_ocr(self, image, psm=6, clearBorder=False):
        """
        Find 1 license plate and OCR it \n
        Return (lpText, lpCnt)
        """
        # initialize the license plate text
        lpText = None

        # convert the input image to grayscale, locate all candidate
        # license plate regions in the image, and the process the 
        # candidates, leaving us with the actual license plate

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_licence_plate_candidates(gray)
        (lp, lpCnt) = self.locate_license_plate(gray, candidates, clearBorder=clearBorder)

        # only OCR the license plate if the license plate ROI is not 
        # empty

        if lp is not None:
            # OCR the license plate
            options = self.build_tesseract_options(psm)
            lpText=pytesseract.image_to_string(lp, config=options)
            self.debug_imshow("License Plate", lp)
        
        # return a 2-tuple of the OCR'd license plate text along with
        # the contour associated with the license plate region
        return (lpText, lpCnt)