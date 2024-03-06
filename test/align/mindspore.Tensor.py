# PyTorch
import torch

c = [123.675, 116.28, 103.53]



# b = torch.Tensor([10, -5])
# print(torch.Tensor.sum(b))
# print(b.size())
# tensor(5.)

# MindSpore
import mindspore
pixel_mean = mindspore.Tensor(c).view(1, -1, 1, 1)
pixel_mean = mindspore.Tensor(pixel_mean)
# print("pixel_mean", type(pixel_mean), "\n", pixel_mean)


a = mindspore.Tensor([10, -5], mindspore.float32)
# print(a.shape)
# print(a.sum())
# 5.0
# print(a.sum(initial=2))

# print(type(1))


# 7.0
