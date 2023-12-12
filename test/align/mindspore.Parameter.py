# import mindspore
# q = mindspore.Parameter([])
# print(len(q))
# if len(q) == 0:
#     q = mindspore.Parameter([0, 1, 2])
# for i in range(2):
#     x = 1
#     if i == q[1]:
#         print("q[0]",q[0])
#         pass
# # q = sorted([100, 1])
# print(q)

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, One
from mindspore import Parameter
# data = Tensor(np.zeros([1, 2, 3]), mindspore.float32)
data = Tensor([-2], mindspore.float32)
w1 = Parameter(initializer(data, data.shape, mindspore.float32))
w2 = Parameter(data)
w3 = Parameter(initializer(One(), [1, 2, 3], mindspore.float32))
w4 = Parameter(initializer(0, [1, 2, 3], mindspore.float32))
print(w1)
print(w2)