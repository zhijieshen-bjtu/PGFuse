import torch
import torch.nn as nn
from torchsummaryX import summary
from networks import UniFuse, Equi

Net = UniFuse

model = Net(34, 512, 1024,  True, 10, 'cee', True).cuda()

# 输入的示例张量
input_tensor1 = torch.randn(1, 3, 256, 256*6).cuda()
input_tensor2 = torch.randn(1, 3, 256, 256*6).cuda()

# 使用torchsummaryX库计算模型的参数量和FLOPs
summary(model, input_tensor1,input_tensor2)
