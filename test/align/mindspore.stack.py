import numpy as np
from mindspore import Tensor, ops
import mindspore
a = []
input_x1 = Tensor(1, mindspore.float32)
# input_x1 = Tensor(np.array([0, 1]).astype(np.float32))
a.append(input_x1)
print(a)
input_x2 = Tensor(2, mindspore.float32)
output = ops.stack([*(input_x1, input_x2), *(input_x1, input_x2)], 0)
print(output)

# [
# (Tensor(shape=[], dtype=Float32, value= 2), Tensor(shape=[], dtype=Int32, value= 0)),
# (Tensor(shape=[], dtype=Float32, value= 1e-06), Tensor(shape=[], dtype=Int32, value= 0))
# ]