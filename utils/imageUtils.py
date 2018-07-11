import cv2
import os
import numpy as np

def read_a_pic(pic_path, dim, imagewidth, imageheight):
    img = cv2.imread(pic_path)
    img = cv2.resize(img, dsize=(imagewidth, imageheight))
    img_list = []
    img_list.append(img)
    labels = np.zeros([1, dim], dtype=np.float32)
    return img_list, labels