import cv2
import numpy as np 
import os
from .image_utils import ImageUtils

current_directory = os.path.dirname(os.path.realpath(__file__))

inception_path = os.path.join(current_directory, 'model/inception5h')

# Read inception model and classnames from file

class Classifier:
  def __init__(self):
    return

  @staticmethod
  def initialize_dnn():
    class_names_path = os.path.join(inception_path, 'imagenet_comp_graph_label_strings.txt')
    model_path = os.path.join(inception_path, 'tensorflow_inception_graph.pb')

    class_names_descriptor = open(class_names_path, 'r')
    class_names = class_names_descriptor.read().strip().split('\n')

    inception_net = cv2.dnn.readNetFromTensorflow(model_path)

    return inception_net, class_names

  # Create confidence intervals for each bin in class_names for an image
  @staticmethod
  def classify_img(neural_net, img):
    min_confidence = 0.05
    net, class_names = neural_net
    resized = ImageUtils.resize_img(img)

    blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (0,0,0))
    net.setInput(input_blob)

    # Since the net is pretrained, we forward pass for confidence intervals
    predictions = net.forward()

    # !!!!! debug
    print(predictions)

    # Get the indices of the 3 best predictions
    n_best = np.argsort(predictions[0])[::-1][:3]

    for (i, label) in enumerate(n_best):
      is_hotdog_str= "Hotdog" if class_names[label] == "hotdog" else "Not Hotdog"

      img_text = "{}, {:.2f}%".format(is_hotdog_str, predictions[0][label] * 100)