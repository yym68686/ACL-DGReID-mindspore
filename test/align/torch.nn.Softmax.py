import torch.nn as nn
import torch

x = torch.randn(10, 5)

softmax = nn.Softmax(-1)
output = softmax(x)
print(output)

import mindspore.nn as nn
import mindspore
import numpy as np

x = mindspore.Tensor(x.numpy().astype(np.float16))
softmax = nn.Softmax(-1)
output = softmax(x)
print(output)
