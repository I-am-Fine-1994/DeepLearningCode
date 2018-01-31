import tensorflow as tf
import numpy as np

#~ 生成数据
x = np.random.uniform(0, 5, [6, 10, 3])
#~ x = tf.Variable(tf.random_uniform([1, 10]))
#~ x = tf.random_uniform([6, 10, 3], maxval = 5)
#~ print(x)
#~ print("-device:", x.device)
#~ print("-dtyp:", x.dtype)
#~ print("-graph:", x.graph)
#~ print("-name:", x.name)
#~ print("-op:", x.op)
#~ print("-shape:", x.shape)
#~ print("-value_index:", x.value_index)

#~ 实际值，也是预期的输出
y = 3*x + 10

#~ 初始化参数
W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
#~ print(W)
#~ print(b)

#~ 模型预测值
y_ = W*x + b

#~ 创建会话，以便对变量值进行输出
sess = tf.Session()
#~ print(sess.run(x))
#~ print(sess.run(y))

#~ 初始化变量，对变量进行输出
init = tf.global_variables_initializer()
sess.run(init)
#~ print(sess.run(W))
#~ print(sess.run(b))

#~ print(sess.run(y_))

#~ 计算损失
loss = tf.losses.mean_squared_error(y, y_)
#~ print(sess.run(loss))

#~ 创建优化函数
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

#~ 开始训练
for i in range(500):
    sess.run(train)
    if i%50 is 0:
        print(sess.run([W, b, loss]))
		
sess.close()
