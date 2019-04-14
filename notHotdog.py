import cv2
import numpy
from classify_hotdogs import *

img = cv2.imread('./images/wiener.jpg')

neural_net = Classifier.initialize_dnn()

classified_img = Classifier.classify_img(neural_net, img)

cv2.imshow("classified", classified_img)

cv2.waitKey(0)
