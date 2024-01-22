from mindspore import nn
from mindspore import Tensor
import numpy as np
import mindspore

# # 创建输入数据
# input_data = Tensor(np.ones([8, 3, 256, 128]), dtype=mindspore.float32)

# # 创建卷积层，将通道数从3增加到2048
# conv = nn.Conv2d(3, 2048, kernel_size=3, stride=1, padding=1, pad_mode="pad")
# output = conv(input_data)

# # 创建转置卷积层，将高度和宽度从(256, 128)增加到(16, 8)
# deconv = nn.Conv2dTranspose(2048, 2048, kernel_size=(32, 16), stride=(16, 8))
# print(output.shape)
# output = deconv(output)

import mindspore
import mindspore.nn as nn
from mindspore import Tensor

class MyNet(nn.Cell):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, pad_mode="pad")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1, pad_mode="pad")
        self.pool2 = nn.MaxPool2d(kernel_size=8, stride=8)

    def construct(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x

# 假设输入tensor
input_tensor = Tensor(np.random.randn(8, 3, 256, 128), mindspore.float32)

# 创建网络实例
net = MyNet()

# 前向传播
output_tensor = net(input_tensor)

print(output_tensor.shape)