import tensorflow as tf
import numpy as np
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#~ 通过TFRecords文件实现随机抽样训练和批量化（对一维数据）
x = np.random.uniform(0, 5, [10])
y = 3*x + 10

#~ 将数据写入TFrecords文件中
def convert_to(x, y, name):
    #~ 指定文件名
    filename = os.path.join(os.getcwd(), name+".tfrecords")
    print("Writing", filename)
    #~ 创建写入器对象
    writer = tf.python_io.TFRecordWriter(filename)
    print(x[0].dtype)

    #~ 开始写入，写入时需要注意，一次只能写入一个值，而不能写入一个数组（list）
    #~ 由于这里的数据类型只有一维，所以不需要事先经过任何处理即可写入
    for index in range(len(x)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'x_data': tf.train.Feature(float_list=tf.train.FloatList(value=[x[index]])), 
            'y_data': tf.train.Feature(float_list=tf.train.FloatList(value=[y[index]]))}))
        #~ 写入器写入数据
        writer.write(example.SerializeToString())
    writer.close()

convert_to(x, y, "simline3")

#~ 读取和解析数据
def read_and_decode(filename):
    #~ 想要读取数据需要先将文件名转为一个队列类型
    filename_queue = tf.train.string_input_producer([filename])
    #~ 创建读取器对象
    reader = tf.TFRecordReader()
    #~ 读取器读取，返回(key, value)对，key个人猜测为文件名，value为文件中的内容
    _, se_exp = reader.read(filename_queue)
    #~ 通过对文件内容进行解析获取其中存储的数据
    #~ 注意：numpy默认的float精度为64位，但文件写入的float类型仅仅只有32位
    #~ 因此解析时需要使用tf.float32进行解析
    features = tf.parse_single_example(se_exp,features={\
        'x_data': tf.FixedLenFeature([], tf.float32),\
        'y_data': tf.FixedLenFeature([], tf.float32)})
	
    #~ 由于解析的数据本身已经是float32类型，因此不需要用cast进行转化
    #~ x_data = tf.cast(features['x_data'], tf.float32)
    #~ y_data = tf.cast(features['y_data'], tf.float32)
    x_data = features['x_data']
    y_data = features['y_data']
    return x_data, y_data

x_data, y_data = read_and_decode("simline3.tfrecords")
x_batch, y_batch = tf.train.shuffle_batch([x_data, y_data], 5, 10, 4)

with tf.Session() as sess:
    #~ 创建协调器管理线程
    coord=tf.train.Coordinator()
    #~ 让文件名进入队列
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    val, l = sess.run([x_batch, y_batch])
    print(val, l)
	
