#也是卷积上了

import torch
import torch.nn.functional as f
ip=torch.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]])

ker=torch.tensor([[1,2,1],[0,1,0],[2,1,0]])

ip=torch.reshape(ip,(1,1,5,5))
ker=torch.reshape(ker,(1,1,3,3))

print(ip.shape)
print(ker.shape)

out=f.conv2d(ip,ker,stride=1)
print(out)

out2=f.conv2d(ip,ker,stride=2)
print(out2)

out3=f.conv2d(ip,ker,stride=2,padding=1)
print(out3)