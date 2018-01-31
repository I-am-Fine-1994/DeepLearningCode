import tensorflow as tf
from readmnist import *

#~ 为了方便使用而设定的变量
width, height, train_num, test_num = 28, 28, 60000, 10000
print(width, height, train_num, test_num, "\n")
#~ 载入训练图像数据集
train_imgs, train_img_height, train_img_width, train_img_num \
    = load_train_image()
#~ 载入测试图像数据集
test_imgs, test_img_height, test_img_width, test_img_num \
    = load_test_image()
#~ 载入训练数据集标签
train_labels, train_label_num = load_train_label()
#~ 载入测试数据集标签
test_labels, test_label_num = load_test_label()

#~ 写入TFRecords函数定义
#~ 将训练数据写入TFRecords
#~ 将测试数据写入TFRecords

#~ 搭建网络模型
