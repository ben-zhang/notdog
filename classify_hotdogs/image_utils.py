import cv2
import numpy as np 

class ImageUtils:
  def __init__(self):
    return

  @staticmethod
  def resize_img(img):
    # resize and pad image to 224x224 for inception to use
    image_dim = 224
    WHITE = [255, 255, 255]

    (height, width, _) = img.shape

    if (height < image_dim and width < image_dim):
      return img

    # ensure one side is of length 224
    scale_factor = image_dim / (height if height > width else width)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    pad_width = 0 if 224 - new_width < 0 else 224 - new_width
    pad_height = 0 if 224 - new_height < 0 else 224 - new_height

    dim = (new_width, new_height)

    resized = cv2.resize(img, dim, cv2.INTER_AREA)

    padded = cv2.copyMakeBorder(resized, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=WHITE)

    return padded
