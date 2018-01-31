import tensorflow as tf
import numpy as np
from readmnist import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

width, height, train_num, test_num = 28, 28, 60000, 10000
#~ 读取数据
img_data, width, height, train_num = list(load_train_image())
img_data = np.reshape(img_data, [60000, 784])
img_data = img_data/255
print("img_data type:", img_data.dtype, "\n")

label, train_num = list(load_train_label())
label = np.reshape(label, [60000, 1])
label = np.array(label, dtype = 'int32')
print("label type:", label.dtype, "\n")

#~ 将label转为OneHot表达
#~ label = tf.one_hot(label, depth=10, on_value=0, off_value=1)
#~ print("change label to onehot:", label)
onehot_label = np.zeros([60000, 10])
for i in range(60000):
    onehot_label[i][label[i]] = 1
print(onehot_label[0])

#~ conv2d的input必须是half或float32类型的4D的tensor，这里将数据转化为float32类型
#~ img_data = img_data.astype("float32")
#~ print("change img_data type to:", img_data.dtype, "\n")

#~ 搭建网络结构
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#~ loss = -tf.reduce_sum(y_*tf.log(y))
loss = tf.reduce_mean(tf.square(y_-y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
gradients = optimizer.compute_gradients(loss)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#~ 创建会话，测试op能否正确运行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(10001):
        index = np.random.choice(60000, 50)
        rand_x = img_data[index]
        rand_y = onehot_label[index]
        sess.run(train, {x: rand_x, y_: rand_y})
        if i%200 is 0:
            print("step: %d, loss: %f" % (i, sess.run(loss, {x: rand_x, y_: rand_y})))
        #~ print(sess.run(accuracy, {x: rand_x, y_: rand_y}))
        
