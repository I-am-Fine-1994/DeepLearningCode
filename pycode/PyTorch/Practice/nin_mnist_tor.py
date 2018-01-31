import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

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

class NIN_MNIST(nn.Module):
    def __init__(self):
        super(NIN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 5, padding=2)
        self.cccp1 = nn.Conv2d(96, 64, 1)
        self.cccp2 = nn.Conv2d(64, 48, 1)
        
        self.conv2 = nn.Conv2d(48, 128, 5, padding=2)
        self.cccp3 = nn.Conv2d(128, 96, 1)
        self.cccp4 = nn.Conv2d(96, 48, 1)
        
        self.conv3 = nn.Conv2d(48, 128, 5, padding=2)
        self.cccp5 = nn.Conv2d(128, 96, 1)
        self.cccp6 = nn.Conv2d(96, 10, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.size())
        x = F.relu(self.cccp1(x))
        # print(x.size())
        x = F.relu(self.cccp2(x))
        # print(x.size())
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # print(x.size())
        x = F.dropout2d(x)
        # print(x.size())
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.cccp3(x))
        x = F.relu(self.cccp4(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = F.dropout2d(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.cccp5(x))
        x = F.relu(self.cccp6(x))
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = F.softmax(x)
        return x

net = NIN_MNIST()
net.cuda()

lr=0.1
momentum = 0.9
weight_decay = 0.0005
batch_size = 50
print(net)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, \
    weight_decay=weight_decay)

start = time.time()
for i in range(1001):
    optimizer.zero_grad()
    
    batch = next_batch(batch_size)
    rand_x = batch[0]
    rand_y = batch[1]
    input = Variable(torch.from_numpy(rand_x.astype('float32')))
    target = Variable(torch.from_numpy(rand_y.astype('float32')))
    input, target = input.cuda(), target.cuda()
    
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if i%200 is 0:
        print("step: %d, loss: %f" \
         % (i, loss.data[0]))
end = time.time()
duration = end - start
print(duration)
