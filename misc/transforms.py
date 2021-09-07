import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import cv2

class Scale(object):
    def __init__(self, height, width, color=(127.5, 127.5, 127.5)):
        self.height = height
        self.width = width
        self.bg_color = color
    def __call__(self, image, labels):
        h, w, _ = image.shape
        img, ratio, padw, padh = self._letterbox(image, self.height, self.width, self.bg_color)

        if (len(labels) > 0):
            labels[:, 0] = ratio * w * labels[:, 0] + padw
            labels[:, 1] = ratio * h * labels[:, 1] + padh

        return img, labels

    def _letterbox(self, img, height, width, color, interpolation=cv2.INTER_LINEAR):  # resize a rectangular image to a padded rectangular 
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height)/shape[0], float(width)/shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh