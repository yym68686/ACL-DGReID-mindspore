import torch
import torch.nn.functional as F

# inputs：输入张量，形状为 (batch_size, in_channels, height, width)，其中 batch_size 表示批次大小，in_channels 表示输入通道数，height 和 width 分别表示输入特征图的高度和宽度。
# updated_weight：卷积核张量，形状为 (out_channels, in_channels, kernel_height, kernel_width)，其中 out_channels 表示输出通道数，in_channels 表示输入通道数，kernel_height 和 kernel_width 分别表示卷积核的高度和宽度。
# updated_bias：偏置张量，形状为 (out_channels,)，其中 out_channels 表示输出通道数。
# stride：步长，可以是一个整数或者一个元组，表示在高度和宽度方向上的步长大小。
# padding：填充，可以是一个整数或者一个元组，表示在输入特征图的四周填充的大小。
# dilation：膨胀，可以是一个整数或者一个元组，表示卷积核中各个元素之间的间隔大小。
# groups：分组，表示将输入通道和输出通道分成多个组，每个组内的通道共享一个卷积核。
# 创建输入张量、卷积核张量、偏置张量
# 定义卷积参数
stride = 1
padding = 1
dilation = 1
groups = 1

in_channels = 3
out_channels = 16
kernel_height = 3
kernel_width = 3
batch_size = 1
if kernel_height == kernel_width:
    kernel_size = kernel_height
else:
    raise ValueError('kernel_height must be equal to kernel_width')


inputs = torch.randn(batch_size, in_channels, 32, 32)
# updated_weight = torch.randn(out_channels, in_channels, kernel_height, kernel_width)
# updated_bias = torch.randn(out_channels)
# 执行卷积操作
conv = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
output = conv(inputs)
print(output.shape)
# print(output)

import mindspore
import mindspore.nn as nn
import numpy as np

# 创建输入张量、卷积核张量、偏置张量
# 输入张量的形状为 (1, 3, 32, 32)，表示批次大小为 1，输入通道数为 3，输入特征图的高度和宽度分别为 32；
# 卷积核张量的形状为 (16, 3, 3, 3)，表示输出通道数为 16，输入通道数为 3，卷积核的高度和宽度分别为 3；
# 偏置张量的形状为 (16,)，表示输出通道数为 16。这些张量将被用于后续的卷积操作。
inputs = mindspore.Tensor(inputs.numpy().astype(np.float32))
# updated_weight = mindspore.Tensor(updated_weight.numpy().astype(np.float32))
# updated_bias = mindspore.Tensor(updated_bias.numpy().astype(np.float32))

# 定义卷积参数
pad_mode = 'pad'

# 创建卷积层
conv = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
# conv.weight = updated_weight
# conv.bias = updated_bias
output = conv(inputs)
print(output.shape)
# print(output)