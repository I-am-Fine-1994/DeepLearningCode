import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from troch.autograd import Variable

class inception(nn.Module):
    def __init__(self, input_channel, out_bn1x1, out_cccp3x3, \
    out_bn3x3, out_cccp5x5, out_bn5x5, out_bnpool):
        super(inception, self).__init__()
        self.branch1x1 = nn.Conv2d(input_channel, out_bn1x1, 1)
        self.cccp3x3 = nn.Conv2d(input_channel, out_cccp3x3, 1)
        self.branch3x3 = nn.Conv2d(out_cccp3x3, out_bn3x3, 3, padding=1)
        self.cccp5x5 = nn.Conv2d(input_channel, out_cccp5x5, 1)
        self.branch5x5 = nn.Conv2d(out_cccp5x5, out_bn5x5, 5, padding=2)
        self.branchpool = nn.Conv2d(input_channel, out_bnpool, 1)
    
    def forward(self, x):
        x1x1 = self.branch1x1(x)
        x3x3 = F.relu(self.cccp3x3(x))
        x3x3 = F.relu(self.branch3x3(x3x3))
        x5x5 = F.relu(self.cccp5x5(x))
        x5x5 = F.relu(self.branch5x5(x5x5))
        xpool = F.max_pool(x, 3, stride=1, padding=1)
        xpool = F.branchpool(xpool)
        #########################
        x = x1x1 x3x3 x5x5 xpool
        return x

class additional_softmax(nn.Module):
    def __init__(self, input_channel, output_1x1=128, output_fc1=1024,\
    output_fc2=1000):
        super(additional_softmax, self).__init__()
        self.input_channel = input_channel
        self.conv1x1 = nn.Conv2d(input_channel, output_1x1, 1)
        self.fc1 = nn.Linear(4*4*input_channel, output_fc1)
        self.fc2 = nn.Linear(output_fc1, output_fc2)
    
    def forward(self, x):
        x = F.avg_pool2d(x, 5, stride=3)
        x = F.relu(self.conv1x1(x))
        x = x.view(-1, 4*4*self.input_channel)
        x = F.relu(self.fc1(x))
        x = F.dropout2d(x, 0.7)
        x = F.softmax(self.fc2(x))

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 192, 3, stride=1, padding=1)
        self.inception3a = inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception(192, 128, 128, 192, 32, 96, 64)
        self.inception4a = inception(480, 192, 96, 208, 16, 48, 64)
        self.addsf0 = additional_softmax(512)
        self.inception4b = inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception(512, 112, 144, 288, 32, 64, 64)
        self.addsf1 = additional_softmax(528)
        self.inception4e = inception(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception(832, 384, 192, 384, 48, 128, 128)
        self.fc1 = nn.Linear(1024, 1000)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception3a(x)
        
