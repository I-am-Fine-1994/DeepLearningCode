import tensorflow as tf
import numpy as np
from readmnist import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

#~ 读取数据
width, height, train_num, test_num = 28, 28, 60000, 10000
img_data, width, height, train_num = list(load_train_image())
img_data = np.reshape(img_data, [60000, 784])
print("img_data type:", img_data.dtype, "\n")

label, train_num = list(load_train_label())
label = np.reshape(label, [60000, 1])
label = np.array(label, dtype = 'int32')
print("label type:", label.dtype, "\n")

onehot_label = np.zeros([60000,10])
for i in range(60000):
    onehot_label[i][label[i]] = 1
print(onehot_label[0])

img_data = img_data/255
print("change img_data type to:", img_data.dtype, "\n")

#~ 搭建网络结构
input_data = tf.placeholder(dtype=tf.float32, shape=(None, 784))
target = tf.placeholder(dtype=tf.float32, shape=(None, 10))

x=tf.reshape(input_data, [-1, 28, 28, 1])
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[6]))
conv1 = tf.nn.relu(\
    tf.nn.conv2d(x, W_conv1, strides=[1,1,1,1], padding="SAME")\
    +b_conv1)
    
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
conv2 = tf.nn.relu(\
    tf.nn.conv2d(pool1, W_conv2, strides=[1,1,1,1], padding="VALID")\
    +b_conv2)

pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

W_fc1 = tf.Variable(tf.truncated_normal(shape=[5*5*16, 120], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[120]))
pool2_flat = tf.reshape(pool2, [-1, 5*5*16])
fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1)+b_fc1)

W_fc2 = tf.Variable(tf.truncated_normal(shape=[120, 84], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[84]))
fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2)+b_fc2)

W_sf = tf.Variable(tf.truncated_normal(shape=[84, 10], stddev=0.1))
b_sf = tf.Variable(tf.constant(0.1, shape=[10]))
y_pre = tf.nn.softmax(tf.matmul(fc2, W_sf)+b_sf)

##################################################
# input_data = tf.placeholder(dtype=tf.float32, shape=(None, 784))
# target = tf.placeholder(dtype=tf.float32, shape=(None, 10))

# x=tf.reshape(input_data, [-1, 28, 28, 1])
# W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))
# b_conv1 = tf.Variable(tf.constant(0., shape=[32]))
# conv1 = tf.nn.relu(\
    # tf.nn.conv2d(x, W_conv1, strides=[1,1,1,1], padding="SAME")\
    # +b_conv1)
    
# pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1))
# b_conv2 = tf.Variable(tf.constant(0., shape=[64]))
# conv2 = tf.nn.relu(\
    # tf.nn.conv2d(pool1, W_conv2, strides=[1,1,1,1], padding="VALID")\
    # +b_conv2)

# pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# W_fc1 = tf.Variable(tf.truncated_normal(shape=[5*5*64, 1024], stddev=0.1))
# b_fc1 = tf.Variable(tf.constant(0., shape=[1024]))
# pool2_flat = tf.reshape(pool2, [-1, 5*5*64])
# fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1)+b_fc1)

# W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 512], stddev=0.1))
# b_fc2 = tf.Variable(tf.constant(0., shape=[512]))
# fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2)+b_fc2)

# fc_drop = tf.nn.dropout(fc, 0.5)

# W_sf = tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=0.1))
# b_sf = tf.Variable(tf.constant(0., shape=[10]))
# y_pre = tf.nn.softmax(tf.matmul(fc2, W_sf)+b_sf)

#####################################################

loss = tf.reduce_mean(tf.square(y_pre-target))
# loss = tf.losses.softmax_cross_entropy(target, y_pre)
optimizer = tf.train.GradientDescentOptimizer(0.1)
gradients = optimizer.compute_gradients(loss)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(target,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

def next_batch(batchsize):
    index = np.random.choice(60000, batchsize)
    rand_x = img_data[index]
    rand_y = onehot_label[index]
    return rand_x, rand_y

#~ 创建会话，测试op能否正确运行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(10001):
        rand_x, rand_y = next_batch(50)
        #~ batch = mnist.train.next_batch(50)
        #~ rand_x = batch[0]
        #~ rand_y = batch[1]
        sess.run(train, {input_data: rand_x, target: rand_y})
        if i%200 is 0:
            print("step: %d, loss: %f" % (i, sess.run(loss, {input_data: rand_x, target: rand_y})))
            #~ print("accuracy", sess.run(accuracy, {input_data: rand_x, target: rand_y}))
            #~ print("gradients", sess.run(gradients[0][0][0][0], {input_data: rand_x, target: rand_y}))
            
