import mindspore.nn
import mindspore
import numpy as np

input = mindspore.Tensor(np.ones(shape=[3, 2, 1, 1]), mindspore.float32)
print(input)
output = input.squeeze(-1).squeeze(-1)
# output = mindspore.ops.squeeze(input)
print(output)