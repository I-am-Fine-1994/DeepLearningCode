import tensorflow as tf
import numpy as np
import os.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#~ 通过TFRecords文件实现随机抽样训练和批量化（对高维度数据）
x = np.random.uniform(0, 5, [100, 4, 4, 2])
y = 3*x + 10

#~ 将数据写入TFrecords文件中
def convert_to(x, y, name):
    width = 4
    height = 4
    depth = 2
    
    #~ 指定文件名
    filename = os.path.join(os.getcwd(), name+".tfrecords")
    print("Writing", filename)
    #~ 创建写入器对象
    writer = tf.python_io.TFRecordWriter(filename)
	
    #~ 开始写入
    for index in range(len(x)):
        x_data = x[index].tostring()
        y_data = y[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={\
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),\
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),\
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),\
            'x_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_data])),\
            'y_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_data]))}))
        writer.write(example.SerializeToString())
    writer.close()
        
convert_to(x, y, "simline4")

#~ 读取和解析数据
def read_and_decode(filename):
    print("Reading", filename)
    #~ 想要读取数据需要先将文件名转为一个队列类型
    filename_queue = tf.train.string_input_producer([filename])
    #~ 创建读取器对象
    reader = tf.TFRecordReader()
    #~ 读取器读取，返回(key, value)对，key个人猜测为文件名，value为文件中的内容
    _, se_exp = reader.read(filename_queue)
    features = tf.parse_single_example(se_exp, features={\
        'x_data': tf.FixedLenFeature([], tf.string),\
        'y_data': tf.FixedLenFeature([], tf.string)})
    print("Decoding", filename)
    #~ 通过对文件内容进行解析获取其中存储的数据
    #~ 注意：numpy默认的float精度为64位，但文件直接写入的float类型仅仅只有32位
    #~ 这里由于将数据转化为了tf.string进行存储，因此每个数据都仍然以64位进行存储
    #~ 所以解析自然也需要使用tf.float64来进行解析
    x_data = tf.decode_raw(features['x_data'], tf.float64)
    x_data = tf.reshape(x_data, [4, 4, 2])
    y_data = tf.decode_raw(features['y_data'], tf.float64)
    y_data = tf.reshape(y_data, [4, 4, 2])
    return x_data, y_data
    
x_data, y_data = read_and_decode("simline4.tfrecords")
x_batch, y_batch = tf.train.shuffle_batch([x_data, y_data], 2, 100, 20)

with tf.Session() as sess:
    #~ 创建协调器管理线程
    coord = tf.train.Coordinator()
    #~ 让文件名进入队列
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    val, l = sess.run([x_batch, y_batch])
    print([val, l])
    
    coord.request_stop()
    coord.join(threads)
