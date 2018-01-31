# 主要变化在于定义了一个 mlpconv 函数，从而简化了网络构建的代码

import tensorflow as tf
import time

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

def weights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
def biases(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))
def conv2d(inputs, weights, biases, strides=[1,1,1,1], padding="SAME", \
    activation=tf.nn.relu):
	return activation(tf.nn.conv2d(inputs, weights, strides, padding)\
    +biases)
def max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"):
    return tf.nn.max_pool(inputs, ksize, strides, padding)

# inputs: 输入
# input_channel: 输入通道数
# output_channel: 输出通道数
# W_kernel_size: 权重矩阵的尺寸
def mlpconv(inputs, input_channel, output_channel, W_kernel_size):
    # 初始化权重
    W_conv1, b_conv1 = weights([W_kernel_size, W_kernel_size, \
        input_channel, output_channel]), biases([output_channel])
    W_cccp2, b_cccp2 = weights([1, 1, output_channel, output_channel]),\
        biases([output_channel])
    W_cccp3, b_cccp3 = weights([1, 1, output_channel, output_channel]),\
        biases([output_channel])
    # 计算输出
    outputs = conv2d(inputs, W_conv1, b_conv1)
    outputs = conv2d(outputs, W_cccp2, b_cccp2)
    outputs = conv2d(outputs, W_cccp3, b_cccp3)
    return outputs
# def mlpconv_initializer(input_channel, output_channel, W_kernel_size):
    # W_conv1, b_conv1 = weights([W_kernel_size, W_kernel_size, \
        # input_channel, ouput_channel]), biases([output_channel])
    # W_cccp2, b_cccp2 = weights([1, 1, output_channel, output_channel]),\
        # biases([output_channel])
    # W_cccp3, b_cccp3 = weights([1, 1, output_channel, output_channel]),\
        # biases([output_channel])
    # return W_conv1, b_conv1, W_cccp2, b_cccp2, W_cccp3, b_cccp3
# def mlpconv(inputs, W_conv1, b_conv1, W_cccp2, b_cccp2, W_cccp3, b_cccp3):
    # outputs = conv2d(inputs, W_conv1, b_conv1)
    # outputs = conv2d(outputs, W_cccp2, b_cccp2)
    # outputs = conv2d(outputs, W_cccp3, b_cccp3)
    # return outputs

input_data = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
target = tf.placeholder(dtype=tf.float32, shape=[None, 10])

x = tf.reshape(input_data, [-1, 28, 28, 1])

x = mlpconv(x, 1, 48, 3)
x = max_pool(x)
x = mlpconv(x, 48, 64, 3)
x = max_pool(x)
x = mlpconv(x, 64, 10, 3)
x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], \
    padding="VALID")

x = tf.reshape(x, [-1, 10])
x = tf.nn.softmax(x)
predict = x

# loss = tf.reduce_mean(tf.square(predict-target))
loss = tf.losses.softmax_cross_entropy(target, predict)
optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
train_step = optimizer.minimize(loss)
correct_pre = tf.equal(tf.argmax(target, 1), tf.argmax(predict, 1))
acc = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

vs = tf.trainable_variables()
print("There are %d trainable_variables in the Graph: " % len(vs))
for v in vs:
    print(v)

start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(601):
        batch = mnist.train.next_batch(50)
        rand_x = batch[0]
        rand_y = batch[1]
        sess.run(train_step, {input_data: rand_x, target: rand_y})
        if i%200 is 0:
            # print("GAP: ", (sess.run(gap, \
                # {input_data: rand_x, target: rand_y})))
            print("tarin step: %d, loss: %f acc: %f" % (i, \
                sess.run(loss, {input_data: rand_x, target: rand_y}), \
                sess.run(acc, {input_data: rand_x, target: rand_y})))
end = time.time()
print(end-start)
