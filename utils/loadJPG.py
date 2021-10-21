import numpy as np
import cv2

def loadJpgColor(img):
    return cv2.imread(img)

def loadJpgGrey(img):
    if isinstance(img, list):
        images = list()
    else:
        return np.asarray(img)

