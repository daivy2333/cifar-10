import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("leg")

dataset=torchvision.datasets.CIFAR10("./dataset", train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
"""
#输入张量
ip=torch.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]])
ip=torch.reshape(ip,(-1,1,5,5))
print(ip)
"""

class k(nn.Module):
    def __init__(self):
        super(k,self).__init__()
        self.maxpo=MaxPool2d(kernel_size=(3,3),ceil_mode=True)

    def forward(self,ip):
        out=self.maxpo(ip)
        return out

mo=k()#实例是必须的
#也是池化上了
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("ip",imgs,step)
    out=mo(imgs)
    writer.add_images("out",out,step)
    step+=1

writer.close()
#基本上是固定流程