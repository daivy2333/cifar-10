import torch
from torch import nn

class mo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+1
        return  output

m=mo()
x=torch.tensor(1.0)
output=m(x)
print(output)
#nn的简单应用