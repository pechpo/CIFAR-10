import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models, transforms
import time

device = ( "cuda" if torch.cuda.is_available() else "cpu")

# ./Dataset/image/img_00001.png
# 数据加载
class CifarDataset(Dataset):
    def __init__(self, name):
        # 添加数据集的初始化内容
        self.name = name
        path = Path("./Dataset")
        path_image = path / "image"
        self.data = []
        self.label = []
        with open(path / (name+".txt"), "r") as f:
            for str in tqdm(f, disable=log):
                if self.name != "testset":
                    str, id = str.split()
                    self.label.append(eval(id))
                else:
                    str = str.strip()
                image = Image.open(path_image / str)
                self.data.append(image)

        self.n = len(self.data)

        #self.transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            ##transforms.RandomPosterize(bits=2),
            #transforms.RandomCrop(32, padding=4),
        #])
        self.transform_train = transforms.Compose([
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        # 添加getitem函数的相关内容
        if self.name == "trainset":
            return self.transform_train(self.data[index]), self.label[index]
        elif self.name == "validset":
            return self.transform_test(self.data[index]), self.label[index]
        else:
            return self.transform_test(self.data[index])

    def __len__(self):
        # 添加len函数的相关内容
        return self.n


# 定义 train 函数
def train():
    validation()
    net.train()
    loss_arr = []
    acc_arr = []
    lr = 0.05
    global cnt
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        print(f"epoch {epoch}: Start Training")
        total_loss = 0

        if cnt == 10:
            cnt = 0
            lr *= 0.5
            #print(f"Adjusting learning rate to {lr}")
        
        # 定义优化器
        optimizer = optim.Adam(net.parameters())
        #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        with tqdm(total=len(train_loader), disable=log) as p:
            for data in train_loader:
                images, labels = data
                # Forward
                images = images.to(device)
                labels = labels.to(device)
                pred = net(images)
                loss = criterion(pred, labels)
                # Backward
                loss.backward()
                # Update
                optimizer.step()
                optimizer.zero_grad()
                p.set_postfix({'loss': loss.item()})
                total_loss += loss.item()
                p.update(1)

        avg_loss = total_loss / len(train_loader)
        loss_arr.append(avg_loss)
        print(f"average loss = {avg_loss}")
        # 模型训练n轮之后进行验证
        if epoch % val_num == 0:
            acc_arr.append(validation())

    print('Finished Training!')
    return loss_arr, acc_arr


# 定义 validation 函数
def validation():
    print("Start Validation")
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(dev_loader, disable=log):
            images, labels = data
            # 在这一部分撰写验证的内容
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)
            total += len(labels)
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", correct / total)
    global best_correct, cnt
    if correct > best_correct:
        cnt = 0
        print("Current best, saving model parameters...")
        torch.save(net.state_dict(), modelPath)
        best_correct = correct
    else:
        cnt += 1
    return correct / total

def plot(loss_arr, acc_arr):
    print(loss_arr)
    print(acc_arr)
    fig, ax = plt.subplots()

    x = np.linspace(0, epoch_num, num=epoch_num)
    ax.plot(x, loss_arr, label='loss') # 作y1 = x 图，并标记此线名为linear
    ax.plot(x, acc_arr, label='accuracy') #作y2 = x^2 图，并标记此线名为quadratic

    ax.set_xlabel('epoch') #设置x轴名称 x label
    ax.set_ylabel('loss/accuracy') #设置y轴名称 y label

    ax.legend() #自动检测要在图例中显示的元素，并且显示

    plt.savefig(savePath / 'result.png', dpi = 300) #图形可视化


# 定义 test 函数
def test():
    print("Generating results...")
    # 将预测结果写入result.txt文件中，格式参照实验1
    net.eval()
    with torch.no_grad(), open(savePath / "result.txt", "w") as f:
        for data in tqdm(test_loader, disable=log):
            images = data
            images = images.to(device)
            pred = net(images)
            print(pred.argmax().item(), file=f)


if __name__ == "__main__":
    
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    # 参数设置
    training = True
    log = True
    savePath = Path("./ResNet18")
    modelPath = savePath / "./model.pth"
    epoch_num = 250
    val_num = 1
    cnt = 0  # 多少次没有validation最优

    print("Start Creating Datasets...")
    # 构建数据集
    if not modelPath.exists() or training:
        train_set = CifarDataset("trainset")
        dev_set = CifarDataset("validset")
    test_set = CifarDataset("testset")

    # 构建数据加载器
    if not modelPath.exists() or training:
        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=128)
        dev_loader = DataLoader(dataset=dev_set, batch_size=128)
    test_loader = DataLoader(dataset=test_set, batch_size=1)

    # 初始化模型对象
    net = models.resnet18(pretrained=False)
    # 由于CIFAR10图片小，将一开始的卷积层调小，并删除一开始的池化层
    net.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.MaxPool2d(1)

    #for param in net.parameters():
        #param.requires_grad = False

    num_ftrs = net.fc.in_features 
    #保持in_features不变，修改out_features=10
    net.fc = nn.Linear(num_ftrs, 10)
    net = net.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    best_correct = 0

    # 模型训练

    if not modelPath.exists() or training:
        loss_arr, acc_arr = train()
        plot(loss_arr, acc_arr)
    else:
        net.load_state_dict(torch.load(modelPath))
        
    # 对模型进行测试，并生成预测结果
    test()

    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
