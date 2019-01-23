# encoding=utf-8

import os
import cv2
import numpy as np

from config import DATASET_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, CAPTCHA_LENGTH, CHAR_SET_LEN
from handle_image import handle_image

X = []
Y = []


# 将二值化后的图片转成向量
def image2vector(img):
    returnVect = np.zeros(IMAGE_WIDTH * IMAGE_HEIGHT)
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            returnVect[IMAGE_WIDTH * i + j] = int(img[i, j])

    return returnVect


def tag2int(tag):
    if tag.isdigit():
        return int(tag)
    else:
        tag = tag.lower()
        return ord(tag) - 97 + 10


# 将标签转为向量
def tag2vector(tag):
    returnVect = np.zeros(CHAR_SET_LEN * CAPTCHA_LENGTH)
    for i in range(CAPTCHA_LENGTH):
        returnVect[CHAR_SET_LEN * i + tag2int(tag[i])] = 1
    return returnVect


# 如果标签存在tag.txt里，使用这个函数获取标签
def opentag():
    with open('tag.txt', 'r') as f:
        l = f.readlines()
        for i in l:
            txt = i.rstrip()
            if len(txt) == 4:
                for j in txt:
                    Y.append(tag2vector(int(j)))


def get_trainsets():
    samples = os.listdir('train')

    for i in range(0, DATASET_SIZE):  # len(samples)
        if i % 100 == 0:
            print('read image %d' % i)
        path = os.path.join('train', samples[i])
        # the image file name is <tag>.png
        tag = path[-8:-4]

        img = cv2.imread(path, 0)
        # 去掉边框后为 64*24

        img = handle_image(img)

        X.append(image2vector(img))
        Y.append(tag2vector(tag))
        # or use opentag() if all tags are save in one file


def get_next_batch(step, batch_size=128):
    start = step % DATASET_SIZE
    end = start + batch_size

    batch_x = X[start:end]
    batch_y = Y[start:end]

    if len(batch_x) < batch_size:
        addition = batch_size - len(batch_x)
        batch_x = batch_x + X[0:addition]
        batch_y = batch_y + Y[0:addition]

    return batch_x, batch_y
