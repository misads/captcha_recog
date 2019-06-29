import os
import numpy as np
import tensorflow as tf
import math

n = 10

times = 100

fonts = ['FreeSans.ttf', 'FreeMonoOblique.ttf']

cmd = ''

tr1 = []

with tf.Session() as sess:

    tr1 = sess.run(tf.truncated_normal([times], 24., 3.))
    tr2 = sess.run(tf.truncated_normal([times], 1., 0.5))

for i in range(times):

    f = np.random.randint(0, 2)
    f = fonts[f]

    l = np.random.randint(0, 6)

    #p = math.fabs(sess.run(tf.truncated_normal([], 0., 2.)))

    p = tr2[i]

    r = np.random.randint(0, 7)

    interval = int(tr1[i])

    o = np.random.randint(0, 11)

    cmd = 'python3 vr_gen.py -f %s -l %d -p %f -r %d -i %d -o %d -n %d' % (f, l, p, r, interval, o, n)

    print('%d/%d: %s' % (i, times, cmd))

    os.system(cmd)
