# PyTorch
import torch

b = torch.Tensor([10, -5])
print(torch.Tensor.sum(b))
print(b.size())
# tensor(5.)

# MindSpore
import mindspore as ms

a = ms.Tensor([10, -5], ms.float32)
print(a.shape)
print(a.sum())
# 5.0
print(a.sum(initial=2))
# 7.0
