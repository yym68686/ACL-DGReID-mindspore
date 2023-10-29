from mindspore.ops import operations as P
import mindspore
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer


import mindspore
import numpy as np
from mindspore import Tensor, ops
input_x = Tensor(np.ones([2, 64]), mindspore.float32)
updated_gamma = Parameter(initializer("ones", 64), name="gamma", requires_grad=True)
updated_beta = Parameter(initializer("zeros", 64), name="beta", requires_grad=True)
# scale = Tensor(np.ones([2]), mindspore.float32)
# bias = Tensor(np.ones([2]), mindspore.float32)
mean = Tensor(np.zeros([64]), mindspore.float32)
variance = Tensor(np.zeros([64]), mindspore.float32)
batch_norm = P.BatchNorm()
output = batch_norm(input_x, updated_gamma, updated_beta, mean, variance)
print(output[0])



# bn_train = P.BatchNorm(is_training=True,
#                             epsilon=1e-5,
#                             momentum=0.9,
#                             data_format='NCHW')

# updated_gamma = Parameter(initializer("ones", 64), name="gamma", requires_grad=True)
# updated_beta = Parameter(initializer("zeros", 64), name="beta", requires_grad=True)
# print(initializer(2, (2, 4)))
# inputs = initializer(2, (2, 64))
# print(type(inputs))
# # inputs = mindspore.Tensor([1, 2], dtype=mindspore.float32)
# result = bn_train(inputs, updated_gamma, updated_beta, None, None)


