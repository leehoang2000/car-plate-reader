import sys
from tensorflow import keras
import cv2
import traceback
import alpr_unconstrained.darknet.python.darknet as dn
from alpr_unconstrained.darknet.python.darknet import detect_image

from alpr_unconstrained.src.keras_utils 			import load_model
from glob 						                    import glob
from os.path 				                    	import splitext, basename
from alpr_unconstrained.src.utils 					import im2single
from alpr_unconstrained.src.keras_utils 			import load_model, detect_lp
from alpr_unconstrained.src.label 					import Shape, writeShapes

import numpy as np
from alpr_unconstrained.darknet.python.darknet  import detect
from alpr_unconstrained.src.label				import dknet_label_conversion
from alpr_unconstrained.src.utils 				import nms


WPOD_NET_PATH = "alpr_unconstrained/data/lp-detector/wpod-net_update1.h5"
OCR_WEIGHTS = 'alpr_unconstrained/data/ocr/ocr-net.weights'
OCR_NETCFG  = 'alpr_unconstrained/data/ocr/ocr-net.cfg'
OCR_DATASET = 'alpr_unconstrained/data/ocr/ocr-net.data'


class ALPRU_UTILS:
    def __init__(self, wpod_net_path=WPOD_NET_PATH):
        self.wpod_net = load_model(wpod_net_path)
        self.ocr_threshold = .4

        ocr_weights = bytes(OCR_WEIGHTS, encoding='utf-8')
        ocr_netcfg  = bytes(OCR_NETCFG, encoding='utf-8')
        ocr_dataset = bytes(OCR_DATASET, encoding='utf-8')

        self.ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
        self.ocr_meta = dn.load_meta(ocr_dataset)
        

    def license_plate_detect(self, image):    
        try:
            lp_threshold = .5
            Ivehicle = image

            ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
            side  = int(ratio*288.)
            bound_dim = min(side + (side%(2**4)),608)

            Llp,LlpImgs,_ = detect_lp(self.wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
            
            print(Llp)
            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                return Ilp*255.
        except:
            traceback.print_exc()
            sys.exit(1)

    def license_plate_ocr(self,image):
        try:
            
            R,(width,height) = detect_image(self.ocr_net, self.ocr_meta, image ,thresh=self.ocr_threshold, nms=None)

            if len(R):

                L = dknet_label_conversion(R,width,height)
                L = nms(L,.45)

                L.sort(key=lambda x: x.tl()[0])
                lp_str = ''.join([chr(l.cl()) for l in L])
                return lp_str

            else:
                print('No characters found')
                return None

        except:
            traceback.print_exc()
            sys.exit(1)