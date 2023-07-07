from mindspore import Tensor
from mindspore import dtype as mstype
x = Tensor([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=mstype.float32)
y = Tensor([[1, 2], [3, 4], [5, 6]], dtype=mstype.int32)
output1 = x.reshape((3, 2))
output2 = x.reshape_as(y)
print(output1)
print(output2)
