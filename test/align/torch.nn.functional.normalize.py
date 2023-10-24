import torch
import torch.nn.functional as F

# 生成一个形状为 (2, 3) 的随机张量
x = torch.Tensor([[-0.5059,  0.2943, -0.8749],
        [ 1.3209,  1.0393,  0.8645]])
# x = torch.randn(2, 2)

# 对张量在第 1 个维度上进行 L2 归一化
x_normalized = F.normalize(x, p=2, dim=1)

print(x)
print(x_normalized)

import mindspore.nn
import mindspore
import numpy as np

x = mindspore.Tensor(x.numpy().astype(np.float32))

l2_normalize = mindspore.ops.L2Normalize(axis=1)
x_normalized = l2_normalize(x)
print(x_normalized)