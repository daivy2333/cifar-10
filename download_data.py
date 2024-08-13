import torchvision
from torch.utils.tensorboard import  SummaryWriter
dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#进行一个数据集的下载
train_set=torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)

writer=SummaryWriter("p10")

for i in range(10):
    img,target=test_set[i]#看不懂，但是要俩值才能指定
    writer.add_image("test_set",img,i)