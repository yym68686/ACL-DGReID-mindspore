# PyTorch 代码
import numpy as np
import torch
from torch import Tensor
import mindspore as ms
from mindspore import ops
# ms.set_context(device_target="CPU")
data = np.empty((1, 2, 3, 4)).astype(np.float32)


ret1 = torch.transpose(Tensor(data), dim0=0, dim1=3)
print(ret1.shape)
ret2 = Tensor(data).permute((3, 2, 0, 1))
print(ret2.shape)


ret = ops.Transpose()(ms.Tensor(data), (3, 2, 1, 0))
print(ret.shape)