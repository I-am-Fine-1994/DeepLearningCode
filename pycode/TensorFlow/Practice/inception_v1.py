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
    return activation(tf.nn.conv2d(inputs, weights, strides, padding) +\
    biases)
def max_pool(inputs, ksize=[1,2,2,1], strides=[1,1,1,1], padding="SAME"):
    return tf.nn.max_pool(inputs, ksize, strides, padding)
def avg_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID"):
    return tf.nn.avg_pool(inputs, ksize, strides, padding)

def inception(inputs, input_channel, output_1x1, output_1x3, output_3x3,\
output_mx1, output_1x5, output_5x5):
    W_1x1, b_1x1 = weights([1, 1, input_channel, output_1x1]), biases([output_1x1])
    W_1x3, b_1x3 = weights([1, 1, input_channel, output_1x3]), \
                   biases([output_1x3])
    W_3x3, b_3x3 = weights([3, 3, output_1x3, output_3x3]), \
                   biases([output_3x3])
    W_1x5, b_1x5 = weights([1, 1, input_channel, output_1x5]), \
                   biases([output_1x5])
    W_5x5, b_5x5 = weights([5, 5, output_1x5, output_5x5]), \
                   biases([output_5x5])
    W_mx1, b_mx1 = weights([1, 1, input_channel, output_mx1]), \
                   biases([output_mx1])
    outputs_1x1 = conv2d(inputs, W_1x1, b_1x1)
    outputs_1x3 = conv2d(inputs, W_1x3, b_1x3)
    outputs_1x5 = conv2d(inputs, W_1x5, b_1x5)
    outputs_m = max_pool(inputs, ksize=[1,3,3,1], strides=[1,1,1,1])
    outputs_3x3 = conv2d(outputs_1x3, W_3x3, b_3x3)
    outputs_5x5 = conv2d(outputs_1x5, W_5x5, b_5x5)
    outputs_mx1 = conv2d(outputs_m, W_mx1, b_mx1)
    outputs = tf.concat([outputs_1x1, outputs_3x3, outputs_5x5, \
    outputs_mx1], 3)
    return outputs

def additional_softmax(inputs, input_channel, ouput_1x1=128, ouput_fc1=1024, \
ouput_fc2=1000):
    W_1x1, b_1x1 = weights([1, 1, input_channel, output_1x1]), \
                   biases([ouput_1x1])
    W_fc1, b_fc1 = weights([4*4*input_channel, output_fc1]),  \
                   biases([ouput_fc1])
    W_fc2, b_fc2 = weights([output_fc1, ouput_fc2]), biases([output_fc2])
    
    outputs = avg_pool(inputs, ksize=[1,5,5,1], strides=[1,3,3,1])
    outputs = tf.nn.relu(conv2d(outputs, W_1x1, b_1x1))
    outputs_flat = tf.reshape(inputs, [-1, 4*4*input_channel])
    outputs = tf.nn.relu(tf.matmul(outputs_flat, W_fc1) + b_fc1)
    outputs = tf.nn.dropout(outputs, 0.7)
    outputs = tf.nn.relu(tf.matmul(outputs, W_fc2) + b_fc2)
    outputs = tf.nn.softmax(outputs)
    return outputs

W_conv1, b_conv1 = weights([7, 7, 3, 64]), biases([64])
W_conv2, b_conv2 = weights([1, 1, 64, 64]), biases([64])
W_conv3, b_conv3 = weights([3, 3, 64, 192]), biases([192])

W_fc_final, b_fc_final = weights([1024, 1000]), biases([1000])
W_sf, b_sf = weights([1000, 1000]), biases([1000])

inputs_data = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
target = tf.placeholder(dtype=tf.float32, shape=[None, 10])

x = tf.reshape(inputs_data, [-1, 28, 28, 1])
x = conv2d(x, W_conv1, b_conv1, strides=[1,2,2,1])
x = max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1])

x = conv2d(x, W_conv2, b_conv2, padding="VALID")
x = conv2d(x, W_conv3, b_conv3)
x = max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1])

x = inception(x, 192, 64, 96, 128, 16, 32, 32)    # 3a
x = inception(x, 256, 128, 128, 192, 32, 96, 64)  # 3b
x = max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1])

x = inception(x, 480, 192, 96, 208, 16, 48, 64)   # 4a
outputs0 = additional_softmax(x, 512)
x = inception(x, 512, 160, 112, 224, 24, 64, 64)  # 4b
x = inception(x, 512, 128, 128, 256, 24, 64, 64)  # 4c
x = inception(x, 512, 112, 144, 288, 32, 64, 64)  # 4d
outputs1 = additional_softmax(x, 528)
x = inception(x, 528, 256, 160, 320, 32, 128, 128)# 4e
x = max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1])

x = inception(x, 832, 256, 160, 320, 32, 128, 128)# 5a
x = inception(x, 832, 384, 192, 384, 48, 128, 128)# 5b
x = avg_pool(x, ksize=[1,7,7,1], strides=[1,1,1,1])

x = tf.reshape(x, [-1, 1024])
x = tf.nn.dropout(x, 0.4)
x = tf.nn.relu(tf.matmul(x, W_fc_final)+b_fc_final)
x = tf.nn.softmax(tf.matmul(x, W_sf)+b_sf)
outputs2 = x
predict = x

# with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    # batch = mnist.train.next_batch(50)
    # batch_x = batch[0]
    # batch_y = batch[1]
    # sess.run(predict, {inputs_data: batch_x, target: batch_y})
