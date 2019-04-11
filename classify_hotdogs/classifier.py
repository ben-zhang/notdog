import cv2
import numpy
import os
from image_utils import ImageUtils

current_directory = os.path.dirname(os.path.realpath(__file__))

inception_path = os.path.join(current_directory, 'model/inception5h')

# Read inception model and classnames from file
def initialize_dnn():
  class_names_path = os.path.join(inception_path, 'imagenet_comp_graph_label_strings.txt')
  model_path = os.path.join(inception_path, 'tensorflow_inception_graph.pb')

  class_names_descriptor = open(class_names_path, 'r')
  class_names = class_names_descriptor.read().strip().split('\n')

  inception_net = cv2.dnn.readNetFromTensorflow(model_path)

  print(class_names)

  return class_names, inception_net

# Create confidence intervals for each bin in class_names for an image
def classify_img(neural_net, image_path):
  class_names, net = neural_net

  img = cv2.imread(image_path)

  resized = resize_img(img)

  inputBlob = 