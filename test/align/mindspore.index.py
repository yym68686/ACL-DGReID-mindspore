import mindspore

import numpy as np

a = mindspore.Tensor(np.ones((1, 3)))
print(a[True].squeeze(0).shape)
