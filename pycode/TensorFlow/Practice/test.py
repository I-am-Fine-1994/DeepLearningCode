#-*- coding:utf-8 -*-  
import tensorflow as tf  
# 生成一个先入先出队列和一个QueueRunner,生成文件名队列  
filenames = ['A.csv', 'B.csv', 'C.csv']  
filename_queue = \
tf.train.string_input_producer(filenames, shuffle=False)
# 定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# 定义Decoder
example, label = \
tf.decode_csv(value, record_defaults=[['null'], ['null']])

example_batch, label_batch = tf.train.shuffle_batch([example,label], \
batch_size=2, capacity=200, min_after_dequeue=100, num_threads=1)
# 运行Graph
with tf.Session() as sess:
	#创建一个协调器，管理线程
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	#~ print(filename_queue)
	#~ print("key:", sess.run(key))
	#~ print("value:", sess.run(value))
	#~ print(example)
	#~ print("example:", sess.run(example))
	#~ print(label)
	#~ print("label", sess.run(label))
	#~ print(example_batch)
	#~ print("example_batch:", sess.run(example_batch))
	#~ print(label_batch)
	#~ print("label_batch:", sess.run(label_batch))
	#启动QueueRunner, 此时文件名队列已经进队。
	for i in range(10):
		e_val,l_val = sess.run([example_batch, label_batch])
		print(e_val,l_val)
		#~ print(example.eval(), label.eval())
	coord.request_stop()
	coord.join(threads)
