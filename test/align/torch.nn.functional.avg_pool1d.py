import torch
import torch.nn.functional as F
import mindspore
import numpy as np
input_tensor = torch.tensor([[[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]]], dtype=torch.float32)

# stride: 窗口的步长。可以是一个单一的数字或者一个元组 (sW,)。默认值是 kernel_size。
output_tensor = F.avg_pool1d(input_tensor, kernel_size=2, stride=2)
print(output_tensor)

input_x = mindspore.Tensor(input_tensor.numpy().astype(np.float32), mindspore.float32)
# stride: 窗口的步长。可以是一个单一的数字或者一个元组 (sW,)。默认值是 1
output = mindspore.ops.avg_pool1d(input_x, kernel_size=2, stride=2)
print(output)