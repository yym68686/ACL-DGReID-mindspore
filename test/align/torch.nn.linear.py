# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(12345)
np.random.seed(12345) 

x = torch.randn(1, 5)
weight = torch.randn(2, 5)  # 2 output features, each with 5 input features
bias = torch.randn(2)  # 2 output features
output = F.linear(x, weight, bias)
print(output)

# MindSpore 
import mindspore
from mindspore import Tensor, nn, context
import numpy as np

# context.set_context(mode=context.GRAPH_MODE)
# context.set_context(device_target="CPU")

np.random.seed(12345) # 与pytorch保持一致
mindspore.set_seed(12345) # 与pytorch保持一致
x = mindspore.Tensor(x.numpy().astype(np.float32))
weight = mindspore.Tensor(weight.numpy().astype(np.float32))
bias = mindspore.Tensor(bias.numpy().astype(np.float32))

net = nn.Dense(5, 2)
# net.weight.set_data(weight)
# net.bias.set_data(bias)
# net.weight.set_data(mindspore.common.initializer.initializer(weight, net.weight.shape, net.weight.dtype))
# net.bias.set_data(mindspore.common.initializer.initializer(bias, net.bias.shape, net.bias.dtype))
# net.weight = weight
# net.bias = bias
output = net(x, weight, bias)
print(output)