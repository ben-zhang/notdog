import cv2
import numpy
from classify_hotdogs import *

img = cv2.imread('./images/wiener.jpg')

Classifier.classify_img(1, img)
