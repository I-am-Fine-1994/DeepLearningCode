#~ 读入数据
import numpy as np
from readmnist import *

width, height, train_num, test_num = 28, 28, 60000, 10000
#~ 读取数据
img_data, width, height, train_num = list(load_train_image())
img_data = np.reshape(img_data, [60000, 1, 28, 28])
img_data = img_data/255
print("img_data type:", img_data.dtype, "\n")

label, train_num = list(load_train_label())
label = np.reshape(label, [60000, 1])
label = np.array(label, dtype = 'int32')
print("label type:", label.dtype, "\n")

#~ 将label转为OneHot表达
onehot_label = np.zeros([60000, 10])
for i in range(60000):
    onehot_label[i][label[i]] = 1

def next_batch(batchsize):
    index = np.random.choice(60000, batchsize)
    rand_x = img_data[index]
    rand_y = onehot_label[index]
    return rand_x, rand_y
