# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# from torch import nn
import mindspore.ops as ops
import mindspore.nn as nn





# class SELayer(nn.Module):
class SELayer(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # output = self.avg_pool(x, tuple(range(len(x.shape)))[-2:]) 才一样，见 torch.nn.AdaptiveAvgPool2d.py
        # self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.fc = nn.SequentialCell(
            nn.Dense(channel, int(channel / reduction), has_bias=False),
            nn.ReLU(),
            nn.Dense(int(channel / reduction), channel, has_bias=False),
            nn.Sigmoid()
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, int(channel / reduction), bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(int(channel / reduction), channel, bias=False),
        #     nn.Sigmoid()
        # )

    # def forward(self, x):
    def construct(self, x):
        b, c, _, _ = x.size()
        # print("ReduceMean x", x.shape, tuple(range(len(x.shape)))[-2:])
        # y = self.avg_pool(x, tuple(range(len(x.shape)))[-2:]).view(b, c)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)