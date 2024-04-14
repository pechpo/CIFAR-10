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

device = ( "cuda" if torch.cuda.is_available() else "cpu")

# ./Dataset/image/img_00001.png
# 数据加载
class CifarDataset(Dataset):
    def __init__(self, name):
        # 添加数据集的初始化内容
        self.hasLabel = False
        if name in ["trainset", "validset"]:
            self.hasLabel = True
        path = Path("./Dataset")
        path_image = path / "image"
        self.data = []
        self.label = []
        with open(path / (name+".txt"), "r") as f:
            for str in tqdm(f):
                if self.hasLabel:
                    str, id = str.split()
                    self.label.append(eval(id))
                else:
                    str = str.strip()
                image = Image.open(path_image / str)
                arr = np.array(image)
                data = torch.Tensor(arr)
                data = data.permute(2, 0, 1)
                self.data.append(data)

        self.n = len(self.data)

    def __getitem__(self, index):
        # 添加getitem函数的相关内容
        if self.hasLabel:
            return self.data[index], self.label[index]
        else:
            return self.data[index]

    def __len__(self):
        # 添加len函数的相关内容
        return self.n


# 构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义模型的网络结构
        self.Linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 2048),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.Dropout(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        # 定义模型前向传播的内容
        x = self.Linear(x)
        return x

# 定义 train 函数
def train():
    # 参数设置
    epoch_num = 10
    val_num = 1
    validation()
    net.train()
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        print(f"epoch {epoch}: Start Training")
        total_loss = 0
        with tqdm(total=len(train_loader)) as p:
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

        print(f"average loss = {total_loss / len(train_loader)}")
        # 模型训练n轮之后进行验证
        if epoch % val_num == 0:
            validation()

    print('Finished Training!')


# 定义 validation 函数
def validation():
    print("Start Validation")
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(dev_loader):
            images, labels = data
            # 在这一部分撰写验证的内容
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)
            total += 1
            correct += (pred.argmax() == labels[0])

    correct = correct.item()
    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", correct / total)
    global best_correct
    if correct > best_correct:
        print("Current best, saving model parameters...")
        torch.save(net.state_dict(), "model.pth")
        best_correct = correct


# 定义 test 函数
def test():
    print("Generating results...")
    # 将预测结果写入result.txt文件中，格式参照实验1
    net.eval()
    with torch.no_grad(), open("result.txt", "w") as f:
        for data in test_loader:
            images = data
            images = images.to(device)
            pred = net(images)
            print(pred.argmax().item(), file=f)


if __name__ == "__main__":
    path = Path("./model.pth")

    print("Start Creating Datasets...")
    # 构建数据集
    if not path.exists():
        train_set = CifarDataset("trainset")
        dev_set = CifarDataset("validset")
    test_set = CifarDataset("testset")

    # 构建数据加载器
    if not path.exists():
        train_loader = DataLoader(dataset=train_set, batch_size=32)
        dev_loader = DataLoader(dataset=dev_set, batch_size=1)
    test_loader = DataLoader(dataset=test_set, batch_size=1)

    # 初始化模型对象
    net = Net().to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(net.parameters())

    best_correct = 0

    # 模型训练

    if not path.exists():
        train()
    else:
        net.load_state_dict(torch.load("model.pth"))
        
    # 对模型进行测试，并生成预测结果
    test()
