import mindspore
from mindspore import Tensor, ops
from mindspore import Parameter
import numpy as np
input_x = Parameter(Tensor(np.array([[1, 2, 3, 4, 5]]), mindspore.int32), name="x")
indices = Tensor(np.array([[2, 4]]), mindspore.int32)
updates = Tensor(np.array([[8, 8]]), mindspore.int32)
axis = 1
reduction = "none"
output = ops.tensor_scatter_elements(input_x, indices, updates, axis, reduction)
# print(output)

input_x = Parameter(Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.int32), name="x")
indices = Tensor(np.array([[2, -1, 2], [0, 2, 1]]), mindspore.int32)
updates = Tensor(np.array([[1, 2, 2], [4, 5, 8]]), mindspore.int32)
axis = 0
reduction = "add"
output = ops.tensor_scatter_elements(input_x, indices, updates, axis, reduction)
print(output)

