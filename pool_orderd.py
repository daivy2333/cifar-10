# 池化标准模板
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 创建一个SummaryWriter实例，用于TensorBoard日志记录
writer = SummaryWriter("leg")

# 加载CIFAR-10数据集，设置为验证集模式，并将图像转换为张量
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
# 创建数据加载器，设置批量大小为64，不打乱数据，不使用多线程加载，丢弃最后一个不完整的批次
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

# 定义一个池化层的神经网络模块
class K(nn.Module):
    def __init__(self):
        super(K, self).__init__()
        # 初始化最大池化层，使用3x3的卷积核，ceil模式
        self.maxpo = MaxPool2d(kernel_size=(3, 3), ceil_mode=True)

    def forward(self, ip):
        # 定义前向传播过程
        out = self.maxpo(ip)
        return out

# 实例化网络模块
model = K()

# 用于记录TensorBoard日志的步数
step = 0

# 遍历数据加载器中的所有数据
for data in dataloader:
    imgs, targets = data
    # 将批次中的图像添加到TensorBoard日志
    writer.add_images("input_images", imgs, step, dataformats='NCHW')
    # 通过模型进行池化操作
    out = model(imgs)
    # 将池化后的图像添加到TensorBoard日志
    writer.add_images("output_images", out, step, dataformats='NCHW')
    # 更新步数
    step += 1

# 关闭SummaryWriter，确保所有日志都已保存
writer.close()