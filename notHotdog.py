import cv2
import numpy
from classify_hotdogs import *

img = cv2.imread('./images/wiener.jpg')
img = ImageUtils.resize_img(img)

cv2.imshow("image", img)
cv2.waitKey(0)