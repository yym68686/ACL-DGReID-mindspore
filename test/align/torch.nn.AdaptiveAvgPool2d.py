import torch
import torch.nn as nn
import mindspore
import numpy as np
import mindspore.ops as ops

# input = np.random.randn(1, 3, 6, 6)
input = np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
[[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
[[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]])


x = mindspore.Tensor(input, mindspore.float32)
print(x)
op = ops.ReduceMean(keep_dims=True)
# op内传入所有维度的最后两维才是一样的
# 比如输入维数 (1, 3, 6, 6) ，那么output = op(x, (2, 3))才和 torch.nn.AdaptiveAvgPool2d(1) 一样
# 比如输入维数 (1, 3, 6) ，那么output = op(x, (1, 2))才和 torch.nn.AdaptiveAvgPool2d(1) 一样
output = op(x, tuple(range(len(x.shape)))[-2:])
print("ops.ReduceMean")
print(output)

input = torch.Tensor(input)

pool = nn.AdaptiveAvgPool2d(1)
output = pool(input)
print("torch.nn.AdaptiveAvgPool2d")
print(output)