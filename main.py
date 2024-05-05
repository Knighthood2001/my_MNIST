import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# 定义超参数
Batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Epochs = 10

# 构建pipeline，对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 下载，加载数据
from torch.utils.data import DataLoader
#   下载数据集
train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

#   加载数据
train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=Batch_size, shuffle=True)

# 搭建网络
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self,x):
        input_size = x.size(0)  # batchsize*1*28*28，拿到batchsize
        x = self.conv1(x)  #输入batchsize*1*28*28，输出batchsize*10*24*24（28-5+1）5是核的大小
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)  # 池化层提取最鲜明的特点。输入：batchsize*10*24*24，输出batchsize*10*12*12
        x = self.conv2(x) # 输入batchsize*10*12*12，输出batchsize*20*10*10 (12-3+1)
        x = F.relu(x)

        x = x.view(input_size, -1) #输入batchsize*20*10*10，输出batchsize*2000
        x = self.fc1(x)  # 输入batchsize*2000，输出batchsize*500
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 定义优化器
model = Digit().to(device)
optimizer = optim.Adam(model.parameters())

# 定义训练
def train_model(model, device, train_loader,optimizer, epoch):
    # 模型训练
    model.train()
    for batch_index, (data, target) in enumerate(train_loader): # 图片数据，标签
        # 部署到device
        data, target = data.to(device), target.to(device)
        # 梯度初始为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target)  # 交叉熵损失，用来多分类
        # 找到概率值最大的下标
        pred = output.max(1, keepdim=True)  # 也可以这样写pred = output.argmax(dim=1)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        # 下面的打印可以不用
        if batch_index% 3000 == 0:
            print("Train Epoch ： {} \t Loss : {: 6f}".format(epoch, loss.item()))

def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:  # 图片数据，标签
            # 部署到device
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的下标
            pred = output.argmax(dim=1)
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test--Average loss: {:.4f},accuracy : {:.3f}\n".format(test_loss, 100*correct/len(test_loader.dataset)))

# 调用
for epoch in range(1,Epochs+1):
    train_model(model, device, train_loader, optimizer, epoch)
    test_model(model, device, test_loader)
