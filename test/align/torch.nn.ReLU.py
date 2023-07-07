import torch
import torch.nn as nn
import numpy as np

x = torch.Tensor(np.array([-1, 2, -3, 2, -1]))
relu = nn.ReLU(inplace=True)
output = relu(x)
print(output)
print(x)


import mindspore.nn as nn
import mindspore

x = mindspore.Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
relu = nn.ReLU()
output = relu(x)
print(output)
print(x)