#@programming_fever
import cv2
import imutils
import numpy as np
import pytesseract
import os
os.environ['DISPLAY'] = ':0'

img = cv2.imread('./sample_images/base.jpg',cv2.IMREAD_COLOR)
img = cv2.resize(img, (600,400) )
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
gray = cv2.bilateralFilter(gray, 13, 15, 15) 
edged = cv2.Canny(gray, 30, 100) 

def locate_licence_plate_candidates(gray, keep=5):
    #Blackhat the image to reveal black characters on white background
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    #Fill small holes and reveal large light regions
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light,0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]

    # Compute Scharr gradient magnitude in x-direction of blackhat
    # scale it back to range [0,255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255*((gradX - minVal)/(maxVal - minVal))
    gradX = gradX.astype("uint8")

    # smooth to group regions that may contain boundaries
    # Apply closing operation (to fill holes) and another binary threshold
    gradX = cv2.GaussianBlur(gradX, (3,3), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # series of erosion and dilations to clean up the thresholded image
    thresh = cv2.erode(thresh, None, iterations = 2)
    thresh = cv2.dilate(thresh, None, iterations = 2)

    # take the bitwise AND between threshold result and the light regions of the image
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    # Find the contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

    # Return the list of contours
    return cnts

# contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)
# contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
contours = locate_licence_plate_candidates(gray)
screenCnt = None

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
cv2.imshow("contour",img)
cv2.waitKey(0)

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    print(len(approx))
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

text = pytesseract.image_to_string(Cropped, config='--psm 6')
print(text)
img = cv2.resize(img,(500,300))
Cropped = cv2.resize(Cropped,(400,200))
cv2.putText(Cropped, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 125, 255), 2)

cv2.imshow('car',img)
cv2.imshow('Cropped',Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()