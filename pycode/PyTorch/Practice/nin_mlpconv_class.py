# 2017年12月04日11:21:14
# 新增 mlpconv 类简化代码
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from import_mnist import next_batch

class mlpconv(nn.Module):
    def __init__(self, input_channel, conv_size, conv_out, cccp1_out, \
    cccp2_out, padding):
        super(mlpconv, self).__init__()
        if padding == "SAME":
            pad = conv_size//2
        if padding == "VALID":
            pad = 0
        self.conv1 = nn.Conv2d(input_channel, conv_out, conv_size, \
            padding=pad)
        self.cccp1 = nn.Conv2d(conv_out, cccp1_out, 1)
        self.cccp2 = nn.Conv2d(cccp1_out, cccp2_out, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.cccp1(x))
        x = F.relu(self.cccp2(x))
        return x

class NIN_MNIST(nn.Module):
    def __init__(self):
        super(NIN_MNIST, self).__init__()
        self.mlpconv1 = mlpconv(1, 5, 96, 64, 48, "SAME")
        self.mlpconv2 = mlpconv(48, 5, 128, 96, 48, "SAME")
        self.mlpconv3 = mlpconv(48, 5, 128, 96, 10, "SAME")
    
    def forward(self, x):
        x = self.mlpconv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = F.dropout2d(x)
        
        x = self.mlpconv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = F.dropout2d(x)
        
        x = self.mlpconv3(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = F.softmax(x)
        return x

if __name__ == "__main__":
    net = NIN_MNIST()
    print(net)
    net.cuda()

    lr=0.1
    momentum = 0.9
    weight_decay = 0.0005
    batch_size = 50
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
