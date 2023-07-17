import torch
import torch.nn.functional as F

# Define input tensor
x = torch.randn(10, 5)  # 10 examples, each with 5 features
weight = torch.randn(2, 5)  # 2 output features, each with 5 input features
bias = torch.randn(2)  # 2 output features
print(bias.shape[0])

output = F.linear(x, weight, bias)
print(output)  # should be (10, 2) representing 10 examples, each with 2 output features

import mindspore.nn as nn
import mindspore
import numpy as np

x = mindspore.Tensor(x.numpy().astype(np.float32))
x = mindspore.Tensor(x)
# x.asnumpy()
weight = mindspore.Tensor(weight.numpy().astype(np.float32))
bias = mindspore.Tensor(bias.numpy().astype(np.float32))

# Define a Dense module with input size of 5 and output size of 2
linear = nn.Dense(5, 2)
linear.weight = weight
linear.bias = bias
output = linear(x)
print(output)  # should be (10, 2) representing 10 examples, each with 2 output features
