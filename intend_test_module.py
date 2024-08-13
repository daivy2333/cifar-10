import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

images_path = "./imgs//OIP-C.jpg"
image = Image.open(images_path).convert('RGB')


transformer = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])

image = transformer(image)

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

    def forward(self, x):
        x = self.module1(x)
        return x
model = k()
model.load_state_dict(torch.load("car10_train_500"))

image = torch.reshape(image , (1, 3, 32, 32))
with torch.no_grad():
    op = model(image)
print(op.argmax(1))

