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

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.flat_num = 7*7*512
        self.classes_num = 10
        self.fc1 = nn.Linear(self.flat_num, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.classes_num)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.view(-1, self.flat_num)
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc2(x)))
        x = F.softmax(self.fc3(x))
        return x

net = VGG16()

lr=0.1
momentum = 0.9
weight_decay = 0.0005
batch_size = 10
print(net)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, \
    weight_decay=weight_decay)

start = time.time()
for i in range(1):
    optimizer.zero_grad()
    
    batch = next_batch(batch_size)
    rand_x = batch[0]
    rand_y = batch[1]
    input = Variable(torch.from_numpy(rand_x.astype('float32')))
    target = Variable(torch.from_numpy(rand_y.astype('float32')))
    
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if i%200 is 0:
        print("step: %d, loss: %f"\
         % (i, loss.data[0]))

end = time.time()
duration = end - start
print(duration)
