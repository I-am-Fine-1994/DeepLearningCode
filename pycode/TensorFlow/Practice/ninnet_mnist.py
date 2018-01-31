# 2017年11月28日16:18:05
# 进一步改写了 mlpconv 函数

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

def mlpconv(inputs, input_channel, kernel_size, conv_output, \
    cccp1_output, cccp2_output):
    # 初始化权重
    W_conv1, b_conv1 = weights([kernel_size, kernel_size, \
        input_channel, conv_output]), biases([conv_output])
    W_cccp2, b_cccp2 = weights([1, 1, conv_output, cccp1_output]),\
        biases([cccp1_output])
    W_cccp3, b_cccp3 = weights([1, 1, cccp1_output, cccp2_output]),\
        biases([cccp2_output])
    # 计算输出
    outputs = conv2d(inputs, W_conv1, b_conv1)
    outputs = conv2d(outputs, W_cccp2, b_cccp2)
    outputs = conv2d(outputs, W_cccp3, b_cccp3)
    return outputs

input_data = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
target = tf.placeholder(dtype=tf.float32, shape=[None, 10])

x = tf.reshape(input_data, [-1, 28, 28, 1])

x = mlpconv(x, 1, 5, 96, 64, 48)
x = max_pool(x)
x = tf.nn.dropout(x, 0.5)
x = mlpconv(x, 48, 5, 128, 96, 48)
x = max_pool(x)
x = tf.nn.dropout(x, 0.5)
x = mlpconv(x, 48, 5, 128, 96, 10)
x = tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], \
    padding="SAME")
x = tf.reshape(x, [-1, 10])
x = tf.nn.softmax(x)
predict = x

# lmb = 0.01
# l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

# loss = tf.reduce_mean(tf.square(predict-target))
loss = tf.losses.softmax_cross_entropy(target, predict)
# optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer = tf.train.MomentumOptimizer(0.1， 0.9)
train_step = optimizer.minimize(loss)
correct_pre = tf.equal(tf.argmax(target, 1), tf.argmax(predict, 1))
acc = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

# config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
start = time.time()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(2001):
        batch = mnist.train.next_batch(50)
        rand_x = batch[0]
        rand_y = batch[1]
        sess.run(train_step, {input_data: rand_x, target: rand_y})
        if i%200 is 0:
            print("tarin step: %d, loss: %f acc: %f" % (i, \
                sess.run(loss, {input_data: rand_x, target: rand_y}), \
                sess.run(acc, {input_data: rand_x, target: rand_y})))
end = time.time()
print(end-start)
