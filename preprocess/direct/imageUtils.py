import cv2
import os
import numpy as np
import imutils
import preprocess.direct.image_preprocess as image_preprocess


def read_a_pic(pic_path, dim, imagewidth, imageheight):
    img = cv2.imread(pic_path)
    img = cv2.resize(img, dsize=(imagewidth, imageheight))
    img_list = []
    img_list.append(img)
    labels = np.zeros([1, dim], dtype=np.float32)
    return img_list, labels


def read_a_pic_reconstruct_slim(pic_path, imageheight, imagewidth):
    img = cv2.imread(pic_path)
    img = image_preprocess.preprocess_for_test(img, imageheight, imagewidth)
    img = imutils.opencv2matplotlib(img)
    img_list = []
    img = list(img)
    img_list.append(img)
    # pic_base = os.path.basename(pic_path)
    return img_list