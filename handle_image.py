# encoding=utf-8
import cv2
import os

from config import IMAGE_HEIGHT, IMAGE_WIDTH


def handle_image(image):

    # 在这里增加代码来对图片进行预处理
    ret, mask = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)  # 二值化





    # 在这里增加代码来对图片进行预处理

    crop = mask[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    return crop


def test_image(dirpath):
    samples = os.listdir(dirpath)

    for i in range(0, 10):  # len(samples)
        path = os.path.join('train', samples[i])
        img = cv2.imread(path, 0)
        img = handle_image(img)

        cv2.imshow('image', img)
        # cv2.imwrite(samples[i],mask)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_image('train')
