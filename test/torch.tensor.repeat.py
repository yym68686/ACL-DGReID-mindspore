import torch
import numpy as np
from mindspore import Tensor
input = np.array([[1, 2],[3, 4], [5, 6]])


x = torch.Tensor(input)
print(x.repeat(2, 2))

x = Tensor(input)
print(x.tile((2, 2)))

# print(x.repeat(2, axis=0).repeat(2, axis=1))
# [] 里面的表示在指定轴上每个不同的元素的重复次数
# print(x.repeat([2, 3, 10], axis=0))
# x = Tensor(np.array(3))
# print(x.repeat(2))