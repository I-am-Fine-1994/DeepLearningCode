import tensorflow as tf
import time

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

# 重写simplecnn2.py

def weights(shape):
	return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
def biases(shape):
	return tf.Variable(tf.constant(0.0, shape=shape))
def conv2d(inputs, weights, biases, strides=[1,1,1,1], padding="SAME"):
	return tf.nn.relu(tf.nn.conv2d(inputs, weights, strides, padding)\
    +biases)
def max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"):
	return tf.nn.max_pool(inputs, ksize, strides, padding)

# W_conv1, b_conv1 = weights([5, 5, 1, 32]), biases([32])
# W_conv2, b_conv2 = weights([5, 5, 32, 64]), biases([64])
# flat_num = 5*5*64
# W_fc1, b_fc1 = weights([5*5*64, 1024]), biases([1024])
# W_fc2, b_fc2 = weights([1024, 512]), biases([512])
# W_fc3, b_fc3 = weights([512, 10]), biases([10])
W_conv1, b_conv1 = weights([5, 5, 1, 6]), biases([6])
W_conv2, b_conv2 = weights([5, 5, 6, 16]), biases([16])
flat_num = 5*5*16
W_fc1, b_fc1 = weights([flat_num, 120]), biases([120])
W_fc2, b_fc2 = weights([120, 84]), biases([84])
W_fc3, b_fc3 = weights([84, 10]), biases([10])

input_data = tf.placeholder(dtype=tf.float32, shape=[None, 784])
target = tf.placeholder(dtype=tf.float32, shape=[None, 10])

x = tf.reshape(input_data, [-1, 28, 28, 1])
# conv1 = conv2d(x, W_conv1, b_conv1)
# pool1 = max_pool(conv1)
# conv2 = conv2d(pool1, W_conv2, b_conv2, padding="VALID")
# pool2 = max_pool(conv2)
# pool2_flat = tf.reshape(pool2, [-1, flat_num])
# fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1) + b_fc1)
# fc1_drop = tf.nn.dropout(fc1, 0.5)
# fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
# fc2_drop = tf.nn.dropout(fc2, 0.5)
# fc3 = tf.nn.softmax(tf.matmul(fc2, W_fc3) + b_fc3)
# predict = fc3

# 仿照PyTorch的写法
x = conv2d(x, W_conv1, b_conv1)
x = max_pool(x)
x = conv2d(x, W_conv2, b_conv2, padding="VALID")
x = max_pool(x)
x = tf.reshape(x, [-1, flat_num])
x = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
x = tf.nn.dropout(x, 0.5)
x = tf.nn.relu(tf.matmul(x, W_fc2) + b_fc2)
x = tf.nn.dropout(x, 0.5)
# x = tf.nn.softmax(tf.matmul(x, W_fc3) + b_fc3)
x = tf.matmul(x, W_fc3) + b_fc3
predict = x

lr = 0.1
# loss = tf.reduce_mean(tf.square(predict-target))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
    logits=predict, labels=target))
optimizer = tf.train.GradientDescentOptimizer(lr)
# optimizer = tf.train.AdadeltaOptimizer()
train_step = optimizer.minimize(loss)

#~ 创建会话，测试op能否正确运行
# config=tf.ConfigProto(log_device_placement=True)
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
