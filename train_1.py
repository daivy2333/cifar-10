import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

test_data=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
class k(nn.Module):
    def __init__(self):
        super(k,self).__init__()
        self.module1=Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)

        )

    def forward(self,x):
        x=self.module1(x)
        return x


mo=k()
loss=nn.CrossEntropyLoss()
optim=torch.optim.SGD(mo.parameters(),lr=0.01)

if __name__=="__main__":
    for epoch in range(20):
        running_loss = 0.0
        for data in dataloader:
            imgs,targets=data
            ops=mo(imgs)

            re_loss=loss(ops,targets)
            optim.zero_grad()
            re_loss.backward()
            optim.step()
            running_loss=running_loss+re_loss
        print(running_loss)