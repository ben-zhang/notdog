import cv2
import os

currentDirectory = os.path.dirname(os.path.realpath(__file__))

inceptionPath = os.path.join(currentDirectory, 'model/inception5h')
# Read inception model and classnames from file
def initializeDNN():
  classNamesPath = os.path.join(inceptionPath, 'imagenet_comp_graph_label_strings.txt')
  modelPath = os.path.join(inceptionPath, 'tensorflow_inception_graph.pb')

  classNamesDescriptor = open(classNamesPath, 'r')
  classNames = classNamesDescriptor.read()

  inceptionNet = cv2.dnn.readNetFromTensorflow(modelPath)

  return classNames, inceptionNet

