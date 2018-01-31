import tensorflow as tf

"""
这份代码并不完整
使用新的方法构建TensorFlow模型，使代码的可读性更强
根据此方法，重写simplecnn2.py为simplecnn_new.py
"""

def weights(shape):
	return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
def biases(shape):
	return tf.Variable(tf.constant(0.0, shape=shape))
def conv2d(inputs, weights, biases, strides=[1,1,1,1], padding="SAME"):
	return tf.nn.relu(tf.nn.conv2d(inputs, weights, strides, padding)\
    +biases)
def max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"):
	return tf.nn.max_pool(inputs, ksize, strides, padding)

# 完全对网络参数取一半
W_conv1, b_conv1 = weights([11, 11, 3, 48]), biases([48])
W_conv2, b_conv2 = weights([5, 5, 48, 128]), biases([128])
W_conv3, b_conv3 = weights([3, 3, 128, 192]), biases([192])
W_conv4, b_conv4 = weights([3, 3, 192, 192]), biases([192])
W_conv5, b_conv5 = weights([3, 3, 192, 128]), biases([128])
flat_num = 13*13*128
W_fc1, b_fc1 = weights([flat_num, 2048]), biases([2048])
W_fc2, b_fc2 = weights([2048, 2048]), biases([2048])
W_fc3, b_fc3 = weights([2048, 1000]), biases([1000])
# 将两个GPU上的参数堆在一个网络中
# W_conv1, b_conv1 = weights([11, 11, 3, 96]), biases([96])
# W_conv2, b_conv2 = weights([5, 5, 96, 256]), biases([256])
# W_conv3, b_conv3 = weights([3, 3, 256, 384]), biases([384])
# W_conv4, b_conv4 = weights([3, 3, 384, 384]), biases([384])
# W_conv5, b_conv5 = weights([3, 3, 384, 256]), biases([256])
# flat_num = 13*13*256
# W_fc1, b_fc1 = weights([flat_num, 4096]), biases([4096])
# W_fc2, b_fc2 = weights([4096, 4096]), biases([4096])
# W_fc3, b_fc3 = weights([4096, 1000]), biases([1000])

conv1 = conv2d(x, W_conv1, b_conv1, strides=[1, 4, 4, 1])
conv2 = conv2d(conv1, W_conv2, b_conv2)
pool2 = max_pool(conv2, ksize=[1, 3, 3, 1])
conv3 = conv2d(pool2, W_conv3, b_conv3)
pool3 = max_pool(conv3, ksize=[1, 3, 3, 1])
conv4 = conv2d(pool3, W_conv4, b_conv4)
conv5 = conv2d(conv4, W_conv5, b_conv5)
pool5 = max_pool(conv5)
pool5_flat = tf.reshape(pool5, [-1, flat_num)
fc1 = tf.nn.relu(tf.matmul(pool5_flat, W_fc1) + b_fc1)
fc1_drop = tf.nn.dropout(fc1, 0.5)
fc2 = tf.nn.relu(tf.matmul(fc1_drop, W_fc2) + b_fc2)
fc2_drop = tf.nn.dropout(fc2, 0.5)
fc3 = tf.nn.softmax(tf.matmul(fc2_drop, W_fc3) + b_fc3)

loss = tf.reduce_mean(tf.square(predict, target))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train_step = optimizer.minimize(loss)
