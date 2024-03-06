import torch
import numpy as np
import torch.nn.functional as F
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn

# inputs：输入张量，形状为 (batch_size, in_channels, height, width)，其中 batch_size 表示批次大小，in_channels 表示输入通道数，height 和 width 分别表示输入特征图的高度和宽度。
# updated_weight：卷积核张量，形状为 (out_channels, in_channels, kernel_height, kernel_width)，其中 out_channels 表示输出通道数，in_channels 表示输入通道数，kernel_height 和 kernel_width 分别表示卷积核的高度和宽度。
# updated_bias：偏置张量，形状为 (out_channels,)，其中 out_channels 表示输出通道数。

# `torch.nn.functional.batch_norm` 是 PyTorch 中用于实现批量归一化（Batch Normalization）的函数之一。
# 批量归一化是一种常用的深度学习技术，旨在加速神经网络的训练，并提高模型的泛化性能。
# 它通过对每个批次的输入数据进行归一化，即通过减去均值并除以标准差来使数据的分布更加稳定，从而减少了网络中每一层的输入数据的分布变化，降低了梯度消失和梯度爆炸的风险。
# `torch.nn.functional.batch_norm` 函数接受输入数据张量（batch）和一组参数，其中包括归一化的均值、方差和缩放和偏移参数。该函数可用于前向传播和反向传播过程中，以帮助训练深度神经网络。
# 创建输入张量、卷积核张量、偏置张量
# 定义卷积参数

in_channels = 3
batch_size = 1

input = torch.randn(batch_size, in_channels, 3, 3)
running_mean = torch.zeros(in_channels)
running_var = torch.ones(in_channels)
updated_weight = torch.ones(in_channels)
updated_bias = torch.zeros(in_channels)
# updated_weight = torch.randn(in_channels)
# updated_bias = torch.randn(in_channels)

training = False
momentum = 0.1
eps = 1e-5
result = F.batch_norm(input, running_mean, running_var, updated_weight, updated_bias, training, momentum, eps)
print(result)
# output = F.batch_norm(input, running_mean, running_var)
# print(output.shape)


# 文档 https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.BatchNorm2d.html?highlight=batchnorm2d#mindspore.nn.BatchNorm2d
input = mindspore.Tensor(input.numpy().astype(np.float32))
# running_mean =  mindspore.Tensor(running_mean.numpy().astype(np.float32))
# running_var =  mindspore.Tensor(running_var.numpy().astype(np.float32))
bias_freeze = False
updated_weight = mindspore.Parameter(mindspore.common.initializer.initializer("ones", updated_weight.shape), name="gamma", requires_grad=True)
updated_bias = mindspore.Parameter(mindspore.common.initializer.initializer("zeros", updated_bias.shape), name="beta", requires_grad=not bias_freeze)

num_features = in_channels
# num_features,
# eps=1e-5,
# momentum=0.9,
# affine=True,
# gamma_init='ones',
# beta_init='zeros',
# moving_mean_init='zeros',
# moving_var_init='ones',
# use_batch_statistics=None,
# data_format='NCHW'
bn = nn.BatchNorm2d(num_features, gamma_init=updated_weight, beta_init=updated_bias, use_batch_statistics=False, momentum=momentum, eps=eps, affine=training)
# bn = nn.BatchNorm2d(num_features, gamma_init=updated_weight, beta_init=updated_bias, use_batch_statistics=True, momentum=momentum, eps=eps, affine=training)
output = bn(input)
# output = ops.batch_norm(input, running_mean, running_var, updated_weight, updated_bias, training, momentum, eps)
print(output)