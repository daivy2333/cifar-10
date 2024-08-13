# 总之环境搭建好了
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 反正是把数据集加载了
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
train_data_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
test_data_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
writer = SummaryWriter("TRAIN")

# 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size)
print(test_data_size)

# 搭建神经网络
from module import *
mo = k()
if torch.cuda.is_available():
    mo = mo.cuda()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.001
optimizer = torch.optim.SGD(mo.parameters(), lr=learning_rate)

# 设置训练网络的参数
to_train_time = 0
to_test_time = 0
train_time = 500

# 训练
for i in range(train_time):
    print(f"第几回训练：{i+1}")

    for data in train_data_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        op = mo(imgs)
        loss = loss_fn(op, targets)

        # 优化器的优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        to_train_time += 1
        if to_train_time % 100 == 0:
            print(f"训次数：{to_train_time},loss:{loss}")
            writer.add_scalar("train_loss", loss, to_train_time)
    to_test_loss = 0
    to_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs,targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            op = mo(imgs)
            loss = loss_fn(op, targets)
            to_test_loss += loss
            accuracy = (op.argmax(1) == targets).sum()
            to_accuracy = to_accuracy + accuracy
    print(f"整体loss：{to_test_loss}")
    print(f"正确率是：{to_accuracy/test_data_size}")
    writer.add_scalar("test_loss", to_test_loss, to_test_time)
    writer.add_scalar("accu",to_accuracy/test_data_size,to_test_time)
    to_test_time += 1
torch.save(mo.state_dict(), "car10_train_500")
print("mo has safed")
writer.close()