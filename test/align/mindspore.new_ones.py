import numpy as np
import mindspore
from mindspore import Tensor
x = Tensor(np.ones((2, 3)), mindspore.float32)
output = x.new_ones(x.shape)
# print(output)
