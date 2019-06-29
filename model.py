import tensorflow as tf
from config import IMAGE_HEIGHT, IMAGE_WIDTH, CAPTCHA_LENGTH, CHAR_SET_LEN, BATCH_SIZE, TRAINING_STEPS

x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
y_ = tf.placeholder(tf.float32, [None, CAPTCHA_LENGTH * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


# define CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x_image = tf.reshape(x, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 3 convelutional layers
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_image, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # fully-connected layer
    # w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    w_d = tf.Variable(w_alpha * tf.random_normal([IMAGE_HEIGHT * IMAGE_WIDTH, 1024]))

    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, 4 * 36]))
    b_out = tf.Variable(b_alpha * tf.random_normal([CAPTCHA_LENGTH * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out
