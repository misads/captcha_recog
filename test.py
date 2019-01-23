# encoding=utf-8
import tensorflow as tf
import numpy as np
import cv2
import os

from handle_image import handle_image
from model import x, keep_prob, crack_captcha_cnn

X = []
Y = []

returnVect = np.zeros(10)
returnVect[1] = 1
for i in range(4):
    Y.append(returnVect)


# 将二值化后的图片转成195维向量
def image2vector(img):
    returnVect = np.zeros(64 * 24)
    for i in range(24):
        for j in range(64):
            returnVect[64 * i + j] = int(img[i, j])

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
    returnVect = np.zeros(36 * 4)
    for i in range(4):
        returnVect[36 * i + tag2int(tag[i])] = 1
    return returnVect


# 这个文件是手动标记的标签，有500行，每行一个4位数，对应训练集
def opentag():
    with open('tag', 'r') as f:
        l = f.readlines()
        for i in l:
            txt = i.rstrip()
            if len(txt) == 4:
                for j in txt:
                    Y.append(tag2vector(int(j)))


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoint'))

        predict = tf.argmax(tf.reshape(output, [-1, 4, 36]), 2)

        text_list = sess.run(predict, feed_dict={x: captcha_image, keep_prob: 1})

        for i in range(0, len(text_list)):
            text = text_list[i].tolist()

            print(ints2tag(text))

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
    for i in range(1, 100):  # len(samples)
        if i % 100 == 0:
            print(i)
        path = os.path.join(dirpath, '%d.png' % i)

        img = cv2.imread(path, 0)
        img = handle_image(img)

        X2.append(image2vector(img))

    crack_captcha(X2)


#predict('test/1.png')
predict_dir('test')
