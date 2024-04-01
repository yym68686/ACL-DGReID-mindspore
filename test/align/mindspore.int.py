import numpy as np
import mindspore
from mindspore import Tensor
input_x = Tensor([True,False], mindspore.bool_)
output = input_x.int()
print(output)
