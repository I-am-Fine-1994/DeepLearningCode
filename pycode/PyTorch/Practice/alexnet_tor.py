import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
import time
from import_mnist import next_batch
# 使用随机生成的数据（因此不关心loss是否有变化，只关心能否运行）
# import numpy as np
# img_data = np.random.choice(255, size=(10, 3, 224, 224))
# img_data = img_data/255
# label = np.random.choice(1000, size=10)
label = np.array(label, dtype = 'int32')
# print("label type:", label[0].dtype)
# 将label转为OneHot表达
# onehot_label = np.zeros([10, 1000])
# for i in range(10):
    # onehot_label[i][label[i]] = 1

# def next_batch(batchsize):
    # index = np.random.choice(10, batchsize)
    # rand_x = img_data[index]
    # rand_y = onehot_label[index]
    # return rand_x, rand_y


class AlexNet(torch.nn.Module):
    
    """对2012年神经网络模型AlexNet的单GPU实现
    由于AlexNet中的LRN被认为没有很大的用处，
    因此PyTorch未提供API，这里去掉了LRN
    """
    
    def __init__(self):
        super(AlexNet, self).__init__()
        # 下面是根据论文对AlexNet进行的实现（取一半参数）
        self.conv1 = torch.nn.Conv2d(3, 48, 11, stride=4, padding=2)
        self.conv2 = torch.nn.Conv2d(48, 128, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(128, 192, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(192, 192, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(192, 128, 3, padding=1)
        self.fc1 = torch.nn.Linear(6*6*128, 2048)
        self.fc2 = torch.nn.Linear(2048, 2048)
        self.fc3 = torch.nn.Linear(2048, 1000)
        # AlexNet完整参数放在一个网络中
        # self.conv1 = torch.nn.Conv2d(3, 96, 11, stride=4)
        # self.conv2 = torch.nn.Conv2d(96, 256, 5)
        # self.conv3 = torch.nn.Conv2d(256, 384, 3)
        # self.conv4 = torch.nn.Conv2d(384, 384, 3)
        # self.conv5 = torch.nn.Conv2d(384, 256, 3)
        # self.fc1 = torch.nn.Linear(13*13*256, 4096)
        # self.fc2 = torch.nn.Linear(4096, 4096)
        # self.fc3 = torch.nn.Linear(4096, 1000)
        # 由于将采用MNIST数据集来测试网络的是否可以运行
        # 因此需要修改一些参数并减少层数（三层卷基层，三层全连接层）
        # 参数的参考来源于：
        # http://www.cnblogs.com/fighting-lady/p/7093217.html
        # self.conv1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=2)
        # self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=2)
        # self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=2)
        # self.fc1 = torch.nn.Linear(4*4*256, 2048)
        # self.fc2 = torch.nn.Linear(2048, 1024)
        # self.fc3 = torch.nn.Linear(1024, 1000)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=3, stride=2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 6*6*128)
        x = F.dropout2d(F.relu(self.fc1(x)))
        x = F.dropout2d(F.relu(self.fc2(x)))
        x = F.softmax(self.fc3(x))
        return x

net = AlexNet()
# net.cuda()
# 对于AlexNet使用的损失函数和反向传播算法不必过于纠结
# 论文中没有提及使用的损失函数，其采用的SGD反向传播算法引入了Momentum
# criterion = torch.nn.MSELoss()
criterion = torch.nn.CrossEntropyLoss()
batch_size = 2
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start = time.time()
for i in range(1):
    optimizer.zero_grad()
    
    batch = next_batch(batch_size)
    rand_x = batch[0]
    rand_y = batch[1]
    input = Variable(torch.from_numpy(rand_x.astype('float32')))
    target = Variable(torch.from_numpy(rand_y.astype('float32')))
    # target = Variable(torch.from_numpy(rand_y.astype('int64')))
    print(target)
    # input, target = input.cuda(), target.cuda()
    
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
