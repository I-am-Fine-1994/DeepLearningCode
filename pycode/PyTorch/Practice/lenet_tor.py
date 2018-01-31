import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from import_mnist import next_batch
import time

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
# net = LeNet()
# print(net)
#~ params = list(net.parameters())

# input = Variable(torch.randn(1, 1, 32, 32))
#~ output = net(input)
#~ print(out)

#~ net.zero_grad()
#~ out.backward(torch.randn(1, 10))

# target = Variable(torch.arange(1, 11))
# criterion = nn.MSELoss()
#~ loss = criterion(output, target)
#~ print(loss)
#~ print(loss.grad_fn)
#~ print(loss.grad_fn.next_functions[0][0])
#~ print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

#~ net.zero_grad()
#~ print("conv1.bias.grad before backward")
#~ print(net.conv1.bias.grad)
#~ loss.backward()
#~ print("conv1.bias.grad after backward")
#~ print(net.conv1.bias.grad)

import torch.optim as optim
# optimizer = optim.SGD(net.parameters(), lr=0.1)

# optimizer.zero_grad()
# output = net(input)
# loss = criterion(output,target)
# loss.backward()
# optimizer.step()
if __name__ == "__main__":
    net = LeNet()
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
