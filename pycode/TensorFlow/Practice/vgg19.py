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

W_conv1, b_conv1 = weights([3, 3, 1, 64]), biases([64])
W_conv2, b_conv2 = weights([3, 3, 64, 64]), biases([64])

W_conv3, b_conv3 = weights([3, 3, 64, 128]), biases([128])
W_conv4, b_conv4 = weights([3, 3, 128, 128]), biases([128])

W_conv5, b_conv5 = weights([3, 3, 128, 256]), biases([256])
W_conv6, b_conv6 = weights([3, 3, 256, 256]), biases([256])
W_conv7, b_conv7 = weights([3, 3, 256, 256]), biases([256])
W_conv8, b_conv8 = weights([3, 3, 256, 256]), biases([256])

W_conv9, b_conv9 = weights([3, 3, 256, 512]), biases([512])
W_conv10, b_conv10 = weights([3, 3, 512, 512]), biases([512])
W_conv11, b_conv11 = weights([3, 3, 512, 512]), biases([512])
W_conv12, b_conv12 = weights([3, 3, 512, 512]), biases([512])

W_conv13, b_conv13 = weights([3, 3, 512, 512]), biases([512])
W_conv14, b_conv14 = weights([3, 3, 512, 512]), biases([512])
W_conv15, b_conv15 = weights([3, 3, 512, 512]), biases([512])
W_conv16, b_conv16 = weights([3, 3, 512, 512]), biases([512])

flat_num = 7*7*512
W_fc1, b_fc1 = weights([flat_num, 4096]), biases([4096])
W_fc2, b_fc2 = weights([4096, 4096]), biases([4096])
W_fc3, b_fc3 = weights([4096, 10]), biases([10])

input_data = tf.placeholder(dtype=tf.float32, shape=[None, 784])
target = tf.placeholder(dtype=tf.float32, shape=[None, 10])

x = tf.reshape(input_data, [-1, 28, 28, 1])

x = conv2d(x, W_conv1, b_conv1)
x = conv2d(x, W_conv2, b_conv2)
# x = max_pool(x)

x = conv2d(x, W_conv3, b_conv3)
x = conv2d(x, W_conv4, b_conv4)
# x = max_pool(x)

x = conv2d(x, W_conv5, b_conv5)
x = conv2d(x, W_conv6, b_conv6)
x = conv2d(x, W_conv7, b_conv7)
x = conv2d(x, W_conv8, b_conv8)
x = max_pool(x)

x = conv2d(x, W_conv9, b_conv9)
x = conv2d(x, W_conv10, b_conv10)
x = conv2d(x, W_conv11, b_conv11)
x = conv2d(x, W_conv12, b_conv12)
# x = max_pool(x)

x = conv2d(x, W_conv13, b_conv13)
x = conv2d(x, W_conv14, b_conv14)
x = conv2d(x, W_conv15, b_conv15)
x = conv2d(x, W_conv16, b_conv16)
x = max_pool(x)

x = tf.reshape(x, [-1, flat_num])
x = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
x = tf.nn.dropout(x, 0.5)
x = tf.nn.relu(tf.matmul(x, W_fc2) + b_fc2)
x = tf.nn.dropout(x, 0.5)
x = tf.nn.softmax(tf.matmul(x, W_fc3) + b_fc3)
predict = x

lr = 0.1
loss = tf.losses.softmax_cross_entropy(target, predict)
optimizer = tf.train.GradientDescentOptimizer(lr)
train_step = optimizer.minimize(loss)

#~ 创建会话，测试op能否正确运行
# config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
start = time.time()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(1001):
        batch = mnist.train.next_batch(50)
        rand_x = batch[0]
        rand_y = batch[1]
        sess.run(train_step, {input_data: rand_x, target: rand_y})
        if i%200 is 0:
            print("step: %d, loss: %f" % (i, sess.run(loss, {input_data: rand_x, target: rand_y})))
end = time.time()
print(end-start)
