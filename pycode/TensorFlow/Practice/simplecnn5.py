import tensorflow as tf
import numpy as np
from readmnist import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#~ 读取数据
width, height, train_num, test_num = 28, 28, 60000, 10000
img_data, width, height, train_num = list(load_train_image())
img_data = np.reshape(img_data, [60000, 784])
print("img_data type:", img_data.dtype, "\n")

label, train_num = list(load_train_label())
label = np.reshape(label, [60000, 1])
label = np.array(label, dtype = 'int32')
print("label type:", label.dtype, "\n")

#~ 将label转为OneHot表达
onehot_label = np.zeros([60000,10])
for i in range(60000):
    onehot_label[i][label[i]] = 1
print(onehot_label[0])

#~ conv2d的input必须是half或float32类型的4D的tensor，这里将数据转化为float32类型
img_data = img_data.astype("float32")
print("change img_data type to:", img_data.dtype, "\n")

x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_actual = tf.placeholder(dtype=tf.float32, shape=[None, 10])

W = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))

y_predict = tf.nn.softmax(tf.matmul(x_input, W)+b)

#~ loss = tf.reduce_mean(
    #~ tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_predict))
#~ loss = -tf.reduce_sum(y_predict*tf.log(y_actual))
loss = tf.reduce_mean(tf.square(y_predict-y_actual))
#~ print(loss)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

def next_batch():
    index = np.random.choice(60000, 50)
    rand_x = img_data[index]
    rand_y = onehot_label[index]
    return rand_x, rand_y

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    rand_x, rand_y = next_batch()
    #~ print(rand_y.shape)
    for i in range(20000):
        rand_x, rand_y = next_batch()
        sess.run(train_step, {x_input: rand_x, y_actual: rand_y})
        if i%1000 is 0:
            print("step: %d, loss: %f" % (i, sess.run(loss, {x_input: rand_x, y_actual: rand_y})))

