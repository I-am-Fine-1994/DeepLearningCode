import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time

# 该网络将运行在GPU上

class CNNModel(nn.Module):
    """这是一个简单的卷积神经网络模型"""
    def __init__(self):
        super(CNNModel, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # self.conv2 = nn.Conv2d(32, 64, 5)
        # self.fc1 = nn.Linear(64*5*5, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 10)
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
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


import torch.optim as optim

net = CNNModel()
net.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.5)

# 开始训练
start = time.time()
for i in range(1):
    optimizer.zero_grad()
    
    #~ 生成batch
    batch = next_batch(50)
    rand_x = batch[0]
    rand_y = batch[1]
    input = Variable(torch.from_numpy(rand_x.astype('float32')))
    target = Variable(torch.from_numpy(rand_y.astype('float32')))
    print(target)
    input, target = input.cuda(), target.cuda()
    
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if i%200 is 0:
        print("step: %d, loss: %f"\
         % (i, loss.data[0]))
end = time.time()
print(end - start)
