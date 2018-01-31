import torch
from torch.autograd import Variable
import torch.nn as nn

class LinearModel(nn.Module):
    """这是一个简单的线性模型"""
    def __init__(self):
        super(LinearModel, self).__init__()
        # 模型的属性包含一个线性操作
        self.lm = nn.Linear(1, 1)
        
    def forward(self, x):
        # 前向传播仅包含一个线性运算
        x = self.lm(x)
        return x
        
import numpy as np

# 利用Numpy生成随机数据作为输入和目标
x = np.random.uniform(size=[10, 1])
y = 3*x + 10
input = Variable(torch.from_numpy(x.astype('float32')))
target = Variable(torch.from_numpy(y.astype('float32')))

import torch.optim as optim

# 创建线性模型实例
lm = LinearModel()

# 创建损失函数实例
criterion = nn.MSELoss()

# 创建优化器实例
optimizer = optim.SGD(lm.parameters(), lr=0.5)

# 开始训练
for i in range(101):
    # 由于PyTorch设计时，梯度会进行累加，因此每次训练都需要先将其置0
    optimizer.zero_grad()
    
    # 计算网络的输出
    output = lm(input)
    
    # 用损失函数实例计算损失
    loss = criterion(output, target)
    
    # 将损失进行反向传播
    loss.backward()
    
    # 优化器会对网络中的参数进行赋值，也就是更新参数
    optimizer.step()
    if i%10 is 0:
        print("step: %d, loss: %f, weight: %f"\
         % (i, loss.data[0], lm.lm.weight.data[0][0]))
