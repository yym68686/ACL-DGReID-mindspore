import torch
import mindspore
import numpy as np

a = torch.Tensor(np.ones((1, 4)))
print(a.data)

b = mindspore.Tensor(np.ones((1, 4)))
print(b.unsqueeze(2))