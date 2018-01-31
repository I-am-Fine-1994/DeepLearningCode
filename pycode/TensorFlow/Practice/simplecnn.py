import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#~ 搭建一个简单的卷积神经玩网络尝试输入数据训练，数据本身没有任何意义，网络结构如下：
#~ 输入数据->卷积层->池化层->卷积层->池化层->全连接层(->全连接层)->Softmax分类器
#~ 随机数据
img_data = np.random.randint(0, 256, size=[10, 28, 28, 1], dtype = 'uint8')
print(img_data[0][0][0].dtype)
#~ 1,1,0,0,1,
#~ 1,1,1,0,1
label = [[0, 1], [0, 1], [1, 0], [1, 0], [0, 1],\
        [0, 1], [0, 1], [0, 1], [1, 0], [0, 1]]
label = np.array(label, dtype = 'int32')
print(label[0].dtype)

#~ conv2d的input必须是half或float32类型的4D的tensor，这里将数据转化为float32类型
img_data = img_data / 255
print(img_data[0][0][0].dtype)

#~ 搭建网络结构
input_data = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
target = tf.placeholder(dtype=tf.float32, shape=(None, 2))

#~ 滤波器矩阵，权重
W_conv1 = tf.Variable(tf.zeros(shape=(3, 3, 1, 2)))
W_conv2 = tf.Variable(tf.zeros(shape=(3, 3, 2, 4)))
W_fc1 = tf.Variable(tf.zeros(shape=(7*7*4, 1024)))
#~ W_fc2 =
W_sf = tf.Variable(tf.zeros(shape=(1024, 2)))

#~ 偏置量
b_conv1 = tf.Variable(tf.zeros(shape=(2)))
b_conv2 = tf.Variable(tf.zeros(shape=(4)))
b_fc1 = tf.Variable(tf.zeros(shape=(1024)))
b_sf = tf.Variable(tf.zeros(shape=(2)))

#~ 第一层卷积层
#~ 由于设定了strides步长均为1，填充设置为相同，但权重矩阵的输出通道为2,
#~ 因此输出的图像大小为10×28×28×2
conv1 = tf.nn.conv2d(input_data, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
#~ 设定第一层卷积的激活函数
h_conv1 = tf.nn.relu(conv1+b_conv1)
#~ 第一层池化层
#~ 由于设定了strides步长均为1，填充设置为不填充
#~ 输出的图像大小为10×14×14×2
pool1 = tf.nn.avg_pool(h_conv1, \
    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
print(pool1)

#~ 第二层卷积层
#~ 输出的图像大小为10×14×14×4
conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME")
#~ 第二层卷积的激活函数
h_conv2 = tf.nn.relu(conv2+b_conv2)
#~ 第二层池化层
#~ 输出的图像大小为10×7×7×4
pool2 = tf.nn.avg_pool(h_conv2,\
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
print(pool2)

#~ 一(两)层全连接层
#~ 在连接之前，先将数据转化为一维
#~ 此处，必须写-1，而不能写None
pool2_flat = tf.reshape(pool2, [-1, 7*7*4])
fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1)+b_fc1)
#~ fc2 =

#~ 连接分类器
y_pre = tf.nn.softmax(tf.matmul(fc1, W_sf)+b_sf)
print(y_pre)

#~ 损失函数
#~ minimize要求输入float32类型的数据，因此这里做一个转换
loss = tf.reduce_mean(tf.square(y_pre-target))
print(loss)
#~ 优化器
optimizer = tf.train.GradientDescentOptimizer(0.01)
#~ 单步训练
train_step = optimizer.minimize(loss)

#~ 创建会话，测试op能否正确运行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(pool1, {input_data: img_data})
    sess.run(pool2, {input_data: img_data})
    sess.run(fc1, {input_data: img_data})
    sess.run(y_pre, {input_data: img_data})
    sess.run(loss, {input_data: img_data, target: label})
    sess.run(train_step, {input_data: img_data, target: label})
