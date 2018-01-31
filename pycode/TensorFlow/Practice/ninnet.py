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

# cccp means cascaded cross channel paramametric pooling
# 每一节代码都是一个 mplconv layer，这里根据论文实现了一个具有三层 mlpconv 的网络
# 问题在于参数没有参考依据，所以效果不是很好
output1 = 48
W_conv1, b_conv1 = weights([3, 3, 1, output1]), biases([output1])
W_cccp2, b_cccp2 = weights([1, 1, output1, output1]), biases([output1])
W_cccp3, b_cccp3 = weights([1, 1, output1, output1]), biases([output1])

output2 = 64
W_conv4, b_conv4 = weights([3, 3, output1, output2]), biases([output2])
W_cccp5, b_cccp5 = weights([1, 1, output2, output2]), biases([output2])
W_cccp6, b_cccp6 = weights([1, 1, output2, output2]), biases([output2])

output3 = 10
W_conv7, b_conv7 = weights([3, 3, output2, output3]), biases([output3])
W_cccp8, b_cccp8 = weights([1, 1, output3, output3]), biases([output3])
W_cccp9, b_cccp9 = weights([1, 1, output3, output3]), biases([output3])

W_fc, b_fc = weights([10, 10]), biases([10])

input_data = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
target = tf.placeholder(dtype=tf.float32, shape=[None, 10])

x = tf.reshape(input_data, [-1, 28, 28, 1])

x = conv2d(x, W_conv1, b_conv1)
x = conv2d(x, W_cccp2, b_cccp2)
x = conv2d(x, W_cccp3, b_cccp3)
x = max_pool(x)

x = conv2d(x, W_conv4, b_conv4)
x = conv2d(x, W_cccp5, b_cccp5)
x = conv2d(x, W_cccp6, b_cccp6)
x = max_pool(x)

x = conv2d(x, W_conv7, b_conv7)
x = conv2d(x, W_cccp8, b_cccp8)
x = conv2d(x, W_cccp9, b_cccp9)

x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], \
    padding="VALID")

x = tf.reshape(x, [-1, 10])
gap = x
x = tf.nn.softmax(x)
predict = x

# loss = tf.reduce_mean(tf.square(predict-target))
loss = tf.losses.softmax_cross_entropy(target, predict)
optimizer = tf.train.MomentumOptimizer(0.1, 0.9)
train_step = optimizer.minimize(loss)
correct_pre = tf.equal(tf.argmax(target, 1), tf.argmax(predict, 1))
acc = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(5001):
        batch = mnist.train.next_batch(50)
        rand_x = batch[0]
        rand_y = batch[1]
        sess.run(train_step, {input_data: rand_x, target: rand_y})
        if i%200 is 0:
            # print("GAP: ", (sess.run(gap, \
                # {input_data: rand_x, target: rand_y})))
            print("tarin step: %d, loss: %f acc: %f" % (i, \
                sess.run(loss, {input_data: rand_x, target: rand_y}),\
                sess.run(acc, {input_data: rand_x, target: rand_y})))
end = time.time()
# print(end-start)
