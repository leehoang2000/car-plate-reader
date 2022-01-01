from anpr import PyImageSearchANPR
from imutils import paths
from dehaze import lowlight_enhance
from dehaze import dehaze
import argparse
import imutils
import cv2
import os
os.environ['DISPLAY'] = ':0'

def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input", required=True, 
    help="path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1, 
    help="whether or not to clear border pixels before OCR")
ap.add_argument("-p","--psm", type=int, default=7,
    help="default PSM mode for OCR'ing license plates")
ap.add_argument("-d","--debug",type=int, default=-1,
    help="whether or not to show additional visualizations")
args = vars(ap.parse_args())

# initialize our ANPR class
anpr = PyImageSearchANPR(debug=args["debug"]>0)

# grab all image paths in the input directory
imagePaths = sorted(list(paths.list_images(args["input"])))

for imagePath in imagePaths:
    # load the input image from disk and resize it
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)

    # low light enhance
    # image = lowlight_enhance(image)

    # dehaze image
    # image = dehaze(image)

    # apply automatic license plate recognition
    (lpText, lpCnt) = anpr.find_and_ocr(image, psm=args["psm"],
        clearBorder=args["clear_border"] > 0)

    if lpText is not None and lpCnt is not None:
        # fit a rotated bounding box to the license plate contour and
        # draw the bounding box on the license plate
        box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

        # Compute a normal (unrotated) bounding box for the license 
        # plate and then draw the OCR'd license plate text on the 
        # image
        (x, y, w, h) = cv2.boundingRect(lpCnt)
        cv2.putText(image, cleanup_text(lpText), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # show the output ANPR image
        print("[INFO] {}".format(lpText))
        cv2.imshow("Output ANPR", image)
        cv2.waitKey(0)
        