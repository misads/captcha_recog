# coding=utf-8
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
import numpy as np
import argparse
import random


def validate_picture(line_num, point_dense, rotate_delta, interval, offset, font):
    total = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345789'
    width = 130
    heighth = 60
    im = Image.new('RGBA', (width, heighth), '#ffffff')  # 设置字体
    font = ImageFont.truetype(font, 40)  # FreeMonoOblique.ttf
    # 创建draw对象

    str = ''
    # 文字
    for item in range(4):
        text = random.choice(total)
        str += text
        draw = ImageDraw.Draw(im)
        draw.text((5 + random.randint(6, 6 + offset) + interval * item, 2 + random.randint(3, 3 + offset)),
                  text=text, fill='#000000', font=font)  # 3377bb

        if rotate_delta:
            im = im.rotate(int(np.random.normal(0, rotate_delta)), resample=2, expand=0)
            fff = Image.new('RGBA', im.size, (255,) * 4)
            # 使用alpha层的rot作为掩码创建一个复合图像
            im = Image.composite(im, fff, im)

        # im.show()
    # 干扰线
    draw = ImageDraw.Draw(im)
    for num in range(line_num):
        x1 = random.randint(0, width / 2)
        y1 = random.randint(0, heighth / 2)
        x2 = random.randint(0, width)
        y2 = random.randint(heighth / 2, heighth)
        draw.line(((x1, y1), (x2, y2)), fill='#000000', width=1)

    # im = im.filter(ImageFilter.FIND_EDGES)
    # im = im.filter(ImageFilter.BLUR)
    # im = im.filter(ImageFilter.SMOOTH)
    # im = im.filter(ImageFilter.SHARPEN)

    for i in range(int(500 * point_dense)):
        x = np.random.randint(0, width)
        y = np.random.randint(0, heighth)
        draw.point((x, y), fill='#000000')

    return im, str


def parse_args():
    # 创建一个parser对象
    parser = argparse.ArgumentParser(description='generate capcha...')
    # parser.add_argument('--net', dest='demo_net', help='',choices=['vgg16', 'res101'], default='res101')
    # parser.add_argument('--save', '-s', action='store_true', help='save')

    parser.add_argument('--num', '-n', default=100, type=int, help='how many capchas to generate, default 100')
    parser.add_argument('--font', '-f', default='FreeSans', type=str,
                        help='font name, default FreeSans.ttf, e.g. FreeMonoOblique.ttf')

    parser.add_argument('--line', '-l', default=5, type=int, help='line num,default 5')
    parser.add_argument('--point', '-p', default=1.0, type=float,
                        help='point dense, default 1.0, recommend to be [0.1, 5]')
    parser.add_argument('--rotate', '-r', default=5, type=int, help='rotate degree(stddev)( recommend to be [0, 10])')
    parser.add_argument('--interval', '-i', default=25, type=int,
                        help='interval between chars,default 25,recommend to be[15, 30]')
    parser.add_argument('--offset', '-o', default=4, type=int,
                        help='max offset, offset will be (a integer) randomly in [0, offset], recommend to be [0, 10])')

    # parser.add_argument('--num', '-n', default=100, type=int, help='how many capchas to generate, default 100')
    # parser.add_argument('--num', '-n', default=100, type=int, help='how many capchas to generate, default 100')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    n = args.num
    font = args.font
    line = args.line
    point = args.point
    rotate = args.rotate
    interval = args.interval
    offset = args.offset

    for i in range(n):
        im, label = validate_picture(line_num=line, point_dense=point, rotate_delta=rotate, interval=interval,
                                     offset=offset, font=font)
        # im.show()
        im.save('./c/' + label + '_%d.jpg' % random.randint(0, 10000))
        #print(label)
