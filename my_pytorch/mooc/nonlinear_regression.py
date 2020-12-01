import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

######################## 准备数据 #########################
# 训练集
train_dataset = datasets.MNIST(root='./',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)
# 测试集
test_dataset = datasets.MNIST(root='./',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)

# 批次大小
batch_size = 64

# 装载训练集
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
# 装载测试集
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True)
# 观察输入和标签的形状（可删去）
for i, data in enumerate(train_loader):
    inputs, labels = data
    print(inputs.shape)
    print(labels.shape)
    break


############################ 搭建网络 ##########################
# 定义网络结构
class net(nn.Module):
    def __init__(self):
        # super().__init__()的作用是执行父类的构造函数，使得我们能够调用父类的属性。
        super(net, self).__init__()
        self.fc = nn.Linear(781, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x
# 定义学习率
lr = 0.5
# 定义模型
model = net()
# 定义代价函数
mse_loss = nn.MSELoss()
# 定义优化器
optimizer = optim.SGD(model.parameters(),lr = lr)
# 打印模型参数（可删去）
for name,parameters in model.named_parameters():
    print('name:{},parameter:{}'.format(name,parameters))
# 定义训练过程
def train():
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        labels = labels.reshape(-1,1)
        one_hot = torch.zeros(inputs.shape[0],10).scatter(1,labels,1)
        loss = mse_loss(out, one_hot)
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 计算梯度
        optimizer.step()
# 定义测试过程
def test():
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum()
    print('Test acc: '.format(correct.item()/len(test_loader)))

for epoch in range(10):
    print('epoch: ', epoch)
    train()
    test()
