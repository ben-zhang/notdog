import cv2
import numpy as np 

class ImageUtils:
  def __init__(self):
    return

  @staticmethod
  def resize_img(img):
    # resize and pad image to 224x224 for inception to use
    image_dim = 224
    white = np.array([255, 255, 255])

    return np.reshape().padToSquare(white)
