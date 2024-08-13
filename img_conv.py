import torch
import torchvision
from torch.utils.data import DataLoader
from torch import  nn
from torch.utils.tensorboard import SummaryWriter
#数据加载
dataset=torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=False,num_workers=0,drop_last=True)

#神经网络
class k(nn.Module):
    def __init__(self):
        super(k,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=(3,3),stride=1,padding=0)
    def forward(self,x):
        x=self.conv1(x)
        return x

mo=k()


writer=SummaryWriter("leg")
step=0
for data in dataloader:
    imgs,targets=data
    output=mo(imgs)
    writer.add_images("in",imgs,step)

    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("out",output,step)


    step+=1

writer.close()
#也是卷积跑起来了