# encoding=utf-8
import tensorflow as tf
import numpy as np
import cv2
import os

from config import IMAGE_WIDTH, IMAGE_HEIGHT, CHAR_SET_LEN, CAPTCHA_LENGTH
from handle_image import handle_image
from model import x, keep_prob, crack_captcha_cnn

X = []
Y = []

returnVect = np.zeros(10)
returnVect[1] = 1
for i in range(CAPTCHA_LENGTH):
    Y.append(returnVect)


# 将二值化后的图片转成195维向量
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


def ints2tag(ints):
    re = ''
    for i in ints:
        if i < 10:
            re = re + str(i)
        else:
            re = re + chr(i - 10 + 97)

    return re


# 将标签转为36*4维向量
def tag2vector(tag):
    returnVect = np.zeros(CHAR_SET_LEN * CAPTCHA_LENGTH)
    for i in range(CAPTCHA_LENGTH):
        returnVect[CHAR_SET_LEN * i + tag2int(tag[i])] = 1
    return returnVect


# 这个文件是手动标记的标签，有500行，每行一个4位数，对应训练集
def opentag():
    with open('tag', 'r') as f:
        l = f.readlines()
        for i in l:
            txt = i.rstrip()
            if len(txt) == CAPTCHA_LENGTH:
                for j in txt:
                    Y.append(tag2vector(int(j)))


def crack_captcha(captcha_image, samples):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoint'))

        predict = tf.argmax(tf.reshape(output, [-1, CAPTCHA_LENGTH, CHAR_SET_LEN]), 2)

        text_list = sess.run(predict, feed_dict={x: captcha_image, keep_prob: 1})

        for i in range(0, len(text_list)):
            text = text_list[i].tolist()

            print(samples[i] + ': ' + ints2tag(text))


def predict(path):
    img = cv2.imread(path, 0)
    img = handle_image(img)
    X2 = []
    X2.append(image2vector(img))
    crack_captcha(X2)


def predict_dir(dirpath):
    X2 = []
    Y2 = []
    samples = os.listdir(dirpath)
    samples.sort()
    for i in samples:  # len(samples)
        path = os.path.join(dirpath, i)

        img = cv2.imread(path, 0)
        img = handle_image(img)

        X2.append(image2vector(img))

    crack_captcha(X2, samples)


#predict('test/1.png')
predict_dir('test')
