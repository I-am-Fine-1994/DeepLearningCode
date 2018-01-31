import tensorflow as tf

def weights(shape):
    return tf.Variable(tf.trucated_normal(shape=shape, stddev=0.1))
def biases(shape):
    return tf.Variable(tf.Constant(0.0, shape=shape))
def conv2d(inputs, weights, biases, strides=[1,1,1,1], padding="SAME",\
    activation=tf.nn.relu):
    return activation(tf.nn.conv2d(inputs, weights, strides, padding)+\
        biases)
def max_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"):
    return tf.nn.max_pool(inputs, ksize, strides, padding)


