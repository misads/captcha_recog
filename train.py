# encoding=utf-8
import tensorflow as tf

from get_batch import get_next_batch, get_trainsets
from config import IMAGE_HEIGHT, IMAGE_WIDTH, CAPTCHA_LENGTH, CHAR_SET_LEN, BATCH_SIZE, TRAINING_STEPS
from model import crack_captcha_cnn, y_, keep_prob, x


def train():
    get_trainsets()

    output = crack_captcha_cnn()

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y_))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, CAPTCHA_LENGTH, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y_, [-1, CAPTCHA_LENGTH, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(step, BATCH_SIZE)
            _, loss_ = sess.run([optimizer, loss], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})


            # calculate accuracy
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(step, 100)
                acc = sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
                print('iterator:%d  loss:%f' % (step, loss_))
                print('accuracy:%f' % acc)
                # save and exit
                if step > TRAINING_STEPS:
                    saver.save(sess, "checkpoint/test")
                    print('saving checkpoint succeeded.')
                    break

            step += 1


train()
