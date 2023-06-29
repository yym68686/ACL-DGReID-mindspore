# PyTorch
import torch
from torch import nn
import numpy as np

net = nn.Linear(3, 4)
x = torch.tensor(np.array([[180, 234, 154], [244, 48, 247]]), dtype=torch.float)
output = net(x)
print(output.detach().numpy())
# (2, 4)

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

net = nn.Dense(3, 4)
x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
output = net(x)
print(output)
# (2, 4)