import tensorflow as tf
import numpy as np
from readmnist import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

width, height, train_num, test_num = 28, 28, 60000, 10000
#~ 读取数据
img_data, width, height, train_num = list(load_train_image())
img_data = np.reshape(img_data, [60000, 28, 28, 1])
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
#~ print(img_data[0])

#~ 搭建网络结构
input_data = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
target = tf.placeholder(dtype=tf.float32, shape=(None, 1))

#~ 滤波器矩阵，权重
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1,32], stddev=0.1))
W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1))
W_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 512], stddev=0.1))
#~ W_fc2 =
W_sf = tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=0.1))

#~ 偏置量
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))
b_sf = tf.Variable(tf.constant(0.1, shape=[10]))

#~ 第一层卷积层
#~ 由于设定了strides步长均为1，填充设置为相同，但权重矩阵的输出通道为2,
#~ 因此输出的图像大小为10×28×28×32
conv1 = tf.nn.conv2d(input_data, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
#~ 设定第一层卷积的激活函数
h_conv1 = tf.nn.relu(conv1+b_conv1)
#~ 第一层池化层
#~ 由于设定了strides步长均为1，填充设置为不填充
#~ 输出的图像大小为10×14×14×32
pool1 = tf.nn.avg_pool(h_conv1, \
    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

#~ 第二层卷积层
#~ 输出的图像大小为10×14×14×64
conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME")
#~ 第二层卷积的激活函数
h_conv2 = tf.nn.relu(conv2+b_conv2)
#~ 第二层池化层
#~ 输出的图像大小为10×7×7×64
pool2 = tf.nn.avg_pool(h_conv2,\
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

#~ 一层全连接层
#~ 在连接之前，先将数据转化为一维
#~ 此处，必须写-1，而不能写None
pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1)+b_fc1)

#~ 连接分类器
y_pre = tf.nn.softmax(tf.matmul(fc1, W_sf)+b_sf)
#~ print(y_pre, "\n")

#~ 损失函数
loss = tf.reduce_mean(tf.square(y_pre-target))
#~ 优化器
optimizer = tf.train.GradientDescentOptimizer(0.1)
gradients = optimizer.compute_gradients(loss)
#~ optimizer = tf.train.AdamOptimizer(0.5)
#~ 单步训练
train_step = optimizer.minimize(loss)

#~ 创建会话，测试op能否正确运行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(500):
        index = np.random.choice(60000, 100)
        rand_x = img_data[index]
        rand_y = label[index]
        sess.run(train_step, {input_data: rand_x, target: rand_y})
        if i%50 is 0 and i is not 0:
            print(sess.run(loss, {input_data: rand_x, target: rand_y}))
