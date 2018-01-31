import torch
from torch.autograd import Variable
import torch.nn as nn

class LinearModel(nn.Module):
    """这是一个简单的线性模型"""
    def __init__(self):
        super(LinearModel, self).__init__()
        # 模型的属性包含一个线性操作
        self.lm = nn.Linear(784, 10)
        
    def forward(self, x):
        # 前向传播仅包含一个线性运算
        x = self.lm(x)
        return x
        
#~ 读入数据
import numpy as np
from readmnist import *

width, height, train_num, test_num = 28, 28, 60000, 10000
#~ 读取数据
img_data, width, height, train_num = list(load_train_image())
img_data = np.reshape(img_data, [60000, 784])
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


import torch.optim as optim

lm = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(lm.parameters(), lr=0.01)

# 开始训练
for i in range(10001):
    optimizer.zero_grad()
    
    #~ 生成batch
    batch = next_batch(50)
    rand_x = batch[0]
    rand_y = batch[1]
    input = Variable(torch.from_numpy(rand_x.astype('float32')))
    target = Variable(torch.from_numpy(rand_y.astype('float32')))
    
    output = lm(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if i%200 is 0:
        print("step: %d, loss: %f"\
         % (i, loss.data[0]))
