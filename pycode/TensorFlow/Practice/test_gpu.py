import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # with tf.device('/gpu:0'):
    a = tf.constant([1., 2.], shape=[1, 2])
    b = tf.constant([2., 3.], shape=[2, 1])
    c = tf.matmul(a, b)
    print(sess.run(c))
