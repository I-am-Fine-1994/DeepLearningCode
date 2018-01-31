import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  
#~ 通过numpy实现对数据的随机抽样训练和批量化
x = np.random.uniform(0, 5, [100, 10, 10])
y = 3*x + 10
batch_size = 50

#~ 创建两个变量节点，为线性模型的两个参数，初始数字随机生成
W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

#~ 构建线性模型，模型的输入由之后随机batch生成的数据提供，因此使用占位符
x_in = tf.placeholder(dtype = tf.float32, shape = (None, 10, 10))
#~ x_in = tf.placeholder(dtype = tf.float32, shape = (None, 1))
#~ x_in = tf.placeholder(dtype = tf.float32, shape = (1, None))
y_pre = W*x_in + b

#~ 计算损失，由相对应的生成的batch提供
y_in = tf.placeholder(dtype = tf.float32, shape = (None, 10, 10))
#~ y_in = tf.placeholder(dtype = tf.float32, shape = (None, 1))
#~ y_in = tf.placeholder(dtype = tf.float32, shape = (1, None))
loss = tf.losses.mean_squared_error(y_in, y_pre)

#~ 创建初始化节点，梯度下降节点
init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

#~ 创建会话，进行训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(500):
        #~ 由numpy生成随机的下标
        index = np.random.choice(100 , batch_size)
        #~ 将对应下标的数据作为一个batch
        rand_x = x[index]
        rand_y = y[index]
        #~ rand_x = np.reshape(rand_x, (1, 6*batch_size))
        #~ rand_y = np.reshape(rand_y, (1, 6*batch_size))
        #~ rand_x = np.reshape(rand_x, (batch_size, 10*10))
        #~ rand_y = np.reshape(rand_y, (batch_size, 10*10))
        #~ 喂数据时，一定要注意保证数据的维度和placeholder的维度完全一致
        sess.run(train, {x_in : rand_x, y_in : rand_y})
        if i%50 is 0:
            #~ print(rand_x)
            #~ print(rand_y)
            print(sess.run([W, b, loss], {x_in : rand_x, y_in : rand_y}))
