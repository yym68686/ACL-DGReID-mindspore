# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math
from collections import OrderedDict

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.autograd.variable import Variable
from fastreid.modeling.ops import MetaConv2d, MetaLinear, MetaBNNorm, MetaINNorm, MetaIBNNorm, MetaGate

from fastreid.layers import (
    IBN,
    SELayer,
    Non_local,
    get_norm,
)
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from fastreid.utils import comm

from mindspore import nn
import mindspore.ops as ops
import mindspore

K = 4
logger = logging.getLogger(__name__)
model_urls = {
    '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ibn_18x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'ibn_34x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'se_ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
}

def repackage_hidden(h):
    if type(h) == mindspore.Tensor:
        return mindspore.Tensor(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# class Sequential_ext(nn.Module):
class Sequential_ext(nn.Cell):
    """A Sequential container extended to also propagate the gating information
    that is needed in the target rate loss.
    """

    def __init__(self, *args):
        super(Sequential_ext, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.insert_child_to_cell(key, module)
        else:
            for idx, module in enumerate(args):
                self.insert_child_to_cell(str(idx), module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._cells.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._cells)

    # def forward(self, input, opt=-1):
    def construct(self, input, opt=-1):
        for i, module in enumerate(self._cells.values()):
            if isinstance(module, MetaConv2d) or isinstance(module, MetaBNNorm):
                input = module(input)
            else:
                input = module(input, opt)
        return input


# class BasicBlock(nn.Module):
class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=False)
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.relu = nn.ReLU()
        # self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    # def forward(self, x):
    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# class MetaSELayer(nn.Module):
class MetaSELayer(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(MetaSELayer, self).__init__()
        # self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Dense(channel, int(channel / reduction), has_bias=False)
        self.fc1 = MetaLinear(channel, int(channel / reduction), has_bias=False)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Dense(int(channel / reduction), channel, has_bias=False)
        self.fc2 = MetaLinear(int(channel / reduction), channel, has_bias=False)
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x, opt=-1):
    def construct(self, x, opt=-1):
        # b, c, _, _ = x.size()
        b, c, _, _ = x.shape
        # print("ReduceMean x", x.shape, tuple(range(len(x.shape)))[-2:], x)
        # y = self.avg_pool(x, tuple(range(len(x.shape)))[-2:]).view(b, c)
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y, opt))
        y = self.sigmoid(self.fc2(y, opt)).view(b, c, 1, 1)

        return x * y.expand_as(x)


class Bottleneck2(nn.Cell):
    expansion = 4*K

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck2, self).__init__()
        self.conv1 = MetaConv2d(inplanes * K, planes, kernel_size=1, bias=False, group=K)
        if with_ibn:
            self.bn1 = MetaIBNNorm(planes)
        else:
            self.bn1 = MetaBNNorm(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False, group=K)
        self.bn2 = MetaBNNorm(planes)
        self.conv3 = MetaConv2d(planes, planes * self.expansion, kernel_size=1, bias=False, group=K)
        self.bn3 = MetaBNNorm(planes * self.expansion)

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    # def forward(self, x, opt=-1):
    def construct(self, x, opt=-1):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x, opt)

        out += residual
        out = self.relu(out)

        return out

# class Bottleneck(nn.Module):
class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        if bn_norm == 'IN':
            norm = MetaINNorm
        else:
            norm = MetaBNNorm
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = MetaIBNNorm(planes)
        else:
            self.bn1 = norm(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = MetaConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU()
        # self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(planes * self.expansion, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    # def forward(self, x, opt=-1):
    def construct(self, x, opt=-1):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # print("ms", mindspore.ops.flatten(out, order='C', start_dim=0, end_dim=-1))
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x, opt)
        out += residual
        out = self.relu(out)

        return out


# class Identity(nn.Module):
class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    # def forward(self, x):
    def construct(self, x):
        return x, None


class HyperRouter(nn.Cell):
    def __init__(self, planes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = ops.ReduceMean(keep_dims=True)
        self.planes = planes
        # self.fc1 = nn.Dense(planes, planes//16)
        # self.fc2 = nn.Dense(planes//16, planes*K)
        # self.fc_classifier = nn.Dense(planes*K, 3)
        self.fc1 = MetaLinear(planes, planes//16)
        self.fc2 = MetaLinear(planes//16, planes*K)
        self.fc_classifier = MetaLinear(planes*K, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    # def forward(self, x, opt=-1):
    def construct(self, x, opt=-1):

        # print("ReduceMean x", x.shape, tuple(range(len(x.shape)))[-2:], x)
        # x = self.avgpool(x, tuple(range(len(x.shape)))[-2:]).squeeze(-1).squeeze(-1)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        # weight = self.relu(F.normalize(self.fc1(x, opt), 2, -1))
        l2_normalize = ops.L2Normalize(axis=-1)
        x_normalized = l2_normalize(self.fc1(x, opt))
        # x_normalized = l2_normalize(self.fc1(x.reshape(-1, self.planes)))

        weight = self.relu(x_normalized)
        # weight = self.fc2(weight).reshape(-1, self.planes, K)
        weight = self.fc2(weight, opt).reshape(-1, self.planes, K)
        # domain_cls_logits = self.fc_classifier(weight.reshape(-1, self.planes*K))
        domain_cls_logits = self.fc_classifier(weight.reshape(-1, self.planes*K), opt)
        x = self.softmax(ops.einsum('bi,bil->bl', x, weight))
        # x = self.softmax(ops.einsum('bi,bil->bl', x.reshape(-1, self.planes), weight))
        # x = self.softmax(torch.einsum('bi,bil->bl', x, weight))

        return x, domain_cls_logits


class ResNet(nn.Cell):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
        self.inplanes = 64
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, pad_mode='pad', padding=3, dilation=1, group=1, has_bias=False)
        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = MetaBNNorm(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0]-1, 1, bn_norm, with_ibn, with_se)

        self.adaptor1_base = block(256, 64, 'IN', False, with_se)
        self.adaptor1_sub = Bottleneck2(256, 64, bn_norm, with_ibn, with_se)
        self.router1 = HyperRouter(256)
        self.invariant_norm1 = MetaBNNorm(256)
        self.specific_norm1 = MetaBNNorm(256)
        self.meta_fuse1 = MetaGate(256)
        self.meta_se1 = MetaSELayer(256)
        self.map1 = MetaBNNorm(256, bias_freeze=True)

        self.layer2 = self._make_layer(block, 128, layers[1]-1, 2, bn_norm, with_ibn, with_se)

        self.adaptor2_base = block(512, 128, 'IN', False, with_se)
        self.adaptor2_sub = Bottleneck2(512, 128, bn_norm, with_ibn, with_se)
        self.router2 = HyperRouter(512)
        self.invariant_norm2 = MetaBNNorm(512)
        self.specific_norm2 = MetaBNNorm(512)
        self.meta_fuse2 = MetaGate(512)
        self.meta_se2 = MetaSELayer(512)
        self.map2 = MetaBNNorm(512, bias_freeze=True)

        self.layer3 = self._make_layer(block, 256, layers[2]-1, 2, bn_norm, with_ibn, with_se)

        self.adaptor3_base = block(1024, 256, 'IN', False, with_se)
        self.adaptor3_sub = Bottleneck2(1024, 256, bn_norm, with_ibn, with_se)
        self.router3 = HyperRouter(1024)
        self.invariant_norm3 = MetaBNNorm(1024)
        self.specific_norm3 = MetaBNNorm(1024)
        self.meta_fuse3 = MetaGate(1024)
        self.meta_se3 = MetaSELayer(1024)
        self.map3 = MetaBNNorm(1024, bias_freeze=True)

        self.layer4 = self._make_layer(block, 512, layers[3]-1, last_stride, bn_norm, with_se=with_se)

        self.adaptor4_base = block(2048, 512, 'IN', False, with_se)
        self.adaptor4_sub = Bottleneck2(2048, 512, bn_norm, with_ibn, with_se)
        self.router4 = HyperRouter(2048)
        self.invariant_norm4 = MetaBNNorm(2048)
        self.specific_norm4 = MetaBNNorm(2048)
        self.meta_fuse4 = MetaGate(2048)
        self.meta_se4 = MetaSELayer(2048)
        self.map4 = MetaBNNorm(2048, bias_freeze=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Standard Params
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.random_init()


        # fmt: off
        # self._build_nonlocal(layers, non_layers, bn_norm)
        if with_nl: self._build_nonlocal(layers, non_layers, bn_norm)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = mindspore.Parameter([-1], requires_grad=False)
            self.NL_1 = self.NL_2 = self.NL_3 = self.NL_4 = nn.SequentialCell(nn.Sigmoid())
        # else:       self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
        # fmt: on

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential_ext(
                MetaConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MetaBNNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        return nn.SequentialCell(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.SequentialCell(
            [Non_local(256, bn_norm) for _ in range(non_layers[0])])
        # self.NL_1 = nn.ModuleList(
        #     [Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = mindspore.Parameter(sorted([layers[0] - (i + 1) for i in range(non_layers[0])]), requires_grad=False)
        # self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.SequentialCell(
            [Non_local(512, bn_norm) for _ in range(non_layers[1])])
        # self.NL_2 = nn.ModuleList(
        #     [Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = mindspore.Parameter(sorted([layers[1] - (i + 1) for i in range(non_layers[1])]), requires_grad=False)
        # self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.SequentialCell(
            [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        # self.NL_3 = nn.ModuleList(/
        #     [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = mindspore.Parameter(sorted([layers[2] - (i + 1) for i in range(non_layers[2])]), requires_grad=False)
        # self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.SequentialCell(
            [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        # self.NL_4 = nn.ModuleList(
        #     [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = mindspore.Parameter(sorted([layers[3] - (i + 1) for i in range(non_layers[3])]), requires_grad=False)
        # self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def get_all_conv_layers(self, module):
        for m in module:
            if isinstance(m, Bottleneck):
                # for _m in m.modules():
                for _m in m.cells_and_names():
                    if isinstance(_m, nn.Conv2d):
                        yield _m

    # def construct(self, x):
    def construct(self, x, epoch, opt=-1):
    # def forward(self, x, epoch, opt=-1):


        # print(4)
        # opt=-1
        # print(f"ndim {x.shape}")
        # print("x type 1", type(x), x)
        x = self.conv1(x)
        # print("x type 2", type(x), x)
        x = self.bn1(x)
        # print("x type 3", type(x), x)
        x = self.relu(x)
        # print("x type 4", type(x))
        x = self.maxpool(x)
        # x = mindspore.Tensor(x)
        # print("x type 5", type(x))

        weights = []
        out_features = []

        # layer 1
        # NL1_counter = 0
        # if len(self.NL_1_idx) == 0:
        #     self.NL_1_idx = mindspore.Parameter([-1])
        #     # self.NL_1_idx = [-1]
        x = self.layer1(x)
        # for i in range(len(self.layer1)):
        #     # x = self.layer1[i](x, opt)
        #     if i == self.NL_1_idx[NL1_counter]:
        #         _, C, H, W = x.shape
        #         # x = self.NL_1[NL1_counter](x)
        #         NL1_counter += 1

        x_invariant = self.adaptor1_base(x, opt)
        # x_invariant = self.adaptor1_base(x.tile((1, K, 1, 1)), opt)
        N, C, H, W = x_invariant.shape
        x_specific = self.adaptor1_sub(x.tile((1, K, 1, 1)), opt).reshape(N, K, C, H, W)
        # x_specific = self.adaptor1_sub(x.repeat(1, K, 1, 1), opt).reshape(N, K, C, H, W)
        weight, domain_cls_logit = self.router1(x, opt)
        weights.append(weight)
        x_specific = (x_specific * weight.reshape(-1, K, 1, 1, 1)).sum(1)
        x_invariant = self.invariant_norm1(x_invariant)
        x_specific = self.specific_norm1(x_specific)
        x = self.meta_fuse1(x_invariant, x_specific, opt)
        x = self.meta_se1(x, opt)
        temp = self.map1(self.avgpool(x))
        l2_normalize = ops.L2Normalize(axis=1)
        x_normalized = l2_normalize(temp)
        out_features.append(x_normalized[..., 0, 0])
        # out_features.append(F.normalize(temp, 2, 1)[..., 0, 0])

        # layer 2
        # NL2_counter = 0
        # if len(self.NL_2_idx) == 0:
        #     self.NL_2_idx = mindspore.Parameter([-1])
        #     # self.NL_2_idx = [-1]
        x = self.layer2(x)
        # for i in range(len(self.layer2)):
        #     # x = self.layer2[i](x, opt)
        #     if i == self.NL_2_idx[NL2_counter]:
        #         _, C, H, W = x.shape
        #         # x = self.NL_2[NL2_counter](x)
        #         NL2_counter += 1

        x_invariant = self.adaptor2_base(x, opt)
        N, C, H, W = x_invariant.shape
        x_specific = self.adaptor2_sub(x.tile((1, K, 1, 1)), opt).reshape(N, K, C, H, W)
        # x_specific = self.adaptor2_sub(x.repeat(1, K, 1, 1), opt).reshape(N, K, C, H, W)
        weight, domain_cls_logit = self.router2(x, opt)
        weights.append(weight)
        x_specific = (x_specific * weight.reshape(-1, K, 1, 1, 1)).sum(1)
        x_invariant = self.invariant_norm2(x_invariant)
        x_specific = self.specific_norm2(x_specific)
        x = self.meta_fuse2(x_invariant, x_specific, opt)
        x = self.meta_se2(x, opt)
        temp = self.map2(self.avgpool(x))
        l2_normalize = ops.L2Normalize(axis=1)
        x_normalized = l2_normalize(temp)
        out_features.append(x_normalized[..., 0, 0])
        # out_features.append(F.normalize(temp, 2, 1)[..., 0, 0])

        # layer 3
        # NL3_counter = 0
        # if len(self.NL_3_idx) == 0:
        #     self.NL_3_idx = mindspore.Parameter([-1])
        #     # self.NL_3_idx = [-1]
        x = self.layer3(x)
        # for i in range(len(self.layer3)):
        #     # x = self.layer3[i](x, opt)
        #     if i == self.NL_3_idx[NL3_counter]:
        #         _, C, H, W = x.shape
        #         # x = self.NL_3[NL3_counter](x)
        #         NL3_counter += 1

        x_invariant = self.adaptor3_base(x, opt)
        N, C, H, W = x_invariant.shape
        x_specific = self.adaptor3_sub(x.tile((1, K, 1, 1)), opt).reshape(N, K, C, H, W)
        weight, domain_cls_logit = self.router3(x, opt)
        weights.append(weight)
        x_specific = (x_specific * weight.reshape(-1, K, 1, 1, 1)).sum(1)
        x_invariant = self.invariant_norm3(x_invariant)
        x_specific = self.specific_norm3(x_specific)
        x = self.meta_fuse3(x_invariant, x_specific, opt)
        x = self.meta_se3(x, opt)
        temp = self.map3(self.avgpool(x))
        l2_normalize = ops.L2Normalize(axis=1)
        x_normalized = l2_normalize(temp)
        out_features.append(x_normalized[..., 0, 0])
        # out_features.append(F.normalize(temp, 2, 1)[..., 0, 0])

        # layer 4
        # NL4_counter = 0
        # if len(self.NL_4_idx) == 0:
        #     self.NL_4_idx = mindspore.Parameter([-1])
        #     # self.NL_4_idx = [-1]
        x = self.layer4(x)
        # for i in range(len(self.layer4)):
        #     # x = self.layer4[i](x, opt)
        #     if i == self.NL_4_idx[NL4_counter]:
        #         _, C, H, W = x.shape
        #         # x = self.NL_4[NL4_counter](x)
        #         NL4_counter += 1

        # print("x.shape", x.shape)
        # print("out_features.len", len(out_features))

        x_invariant = self.adaptor4_base(x, opt)
        N, C, H, W = x_invariant.shape
        x_specific = self.adaptor4_sub(x.tile((1, K, 1, 1)), opt).reshape(N, K, C, H, W)
        weight, domain_cls_logit = self.router4(x, opt)
        weights.append(weight)
        x_specific = (x_specific * weight.reshape(-1, K, 1, 1, 1)).sum(1)
        x_invariant = self.invariant_norm4(x_invariant)
        x_specific = self.specific_norm4(x_specific)
        x = self.meta_fuse4(x_invariant, x_specific, opt)
        x = self.meta_se4(x, opt)
        temp = self.map4(self.avgpool(x))
        l2_normalize = ops.L2Normalize(axis=1)
        x_normalized = l2_normalize(temp)
        out_features.append(x_normalized[..., 0, 0])
        # out_features.append(F.normalize(temp, 2, 1)[..., 0, 0])

        # weights = torch.cat(weights, -1)
        weights = ops.cat(weights, -1)

        # print("x.shape", x.shape)
        # print("out_features.len", len(out_features))
        return x, weights, out_features

    def random_init(self):
        for name, m in self.cells_and_names():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.Normal(sigma=math.sqrt(2. / n), mean=0.0), m.weight.shape, m.weight.dtype))
                # nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(mindspore.common.initializer.initializer("ones", m.gamma.shape, m.gamma.dtype))
                # nn.init.constant_(m.weight, 1)
                m.beta.set_data(mindspore.common.initializer.initializer("zeros", m.beta.shape, m.beta.dtype))
                # nn.init.constant_(m.bias, 0)


def init_pretrained_weights(key):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = model_urls[key].split('/')[-1].split(".pth")[0] + ".ckpt"

    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        logger.info(f"Pretrain model don't exist, downloading from {model_urls[key]}")
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    logger.info(f"Loading pretrained model from {cached_file}")
    # print(cached_file)
    print(1)
    state_dict = mindspore.load_checkpoint(cached_file)
    # state_dict = torch.load(cached_file, map_location=torch.device('cpu'))
    #CHANGE Reduction Version
    # state_dict = torch.load('/home/yuming/.cache/torch/checkpoints/resnet50_ibn_a-d9d0bb7b.pth', map_location=torch.device('cpu'))

    return state_dict


@BACKBONE_REGISTRY.register()
def build_meta_dynamic_router_resnet_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # pretrain      = True
    # pretrain_path = None
    # last_stride   = 1
    # bn_norm       = "BN"
    # with_ibn      = True
    # with_se       = False
    # with_nl       = False
    # depth         = "50x"
    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    with_se       = cfg.MODEL.BACKBONE.WITH_SE
    with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]

    block = {
        '18x': BasicBlock,
        '34x': BasicBlock,
        '50x': Bottleneck,
        '101x': Bottleneck
    }[depth]

    model = ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, nl_layers_per_stage)
    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = mindspore.load_checkpoint(pretrain_path)
                # state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
            if with_ibn: key = 'ibn_' + key
            # if with_se:  key = 'se_' + key

            state_dict = init_pretrained_weights(key)

        # model_dict = model.state_dict()
        model_dict = OrderedDict()
        for item in model.get_parameters():
            model_dict[item.name] = item.value()

        for item in model.get_parameters():
            k = item.name
            if k in state_dict:
                v = state_dict[k]
                if model_dict[k].shape == v.shape:
                    model_dict[k] = v
                else:
                    if len(v.shape) == 1:
                        model_dict[k] = v[:model_dict[k].shape[0]]
                    elif len(v.shape) == 2:
                        model_dict[k] = v[:model_dict[k].shape[0], :model_dict[k].shape[1]]
                    elif len(v.shape) == 3:
                        model_dict[k] = v[:model_dict[k].shape[0], :model_dict[k].shape[1], :model_dict[k].shape[2]]
                    elif len(v.shape) == 4:
                        model_dict[k] = v[:model_dict[k].shape[0], :model_dict[k].shape[1], :model_dict[k].shape[2], :model_dict[k].shape[3]]
                    elif len(v.shape) == 5:
                        model_dict[k] = v[:model_dict[k].shape[0], :model_dict[k].shape[1], :model_dict[k].shape[2], :model_dict[k].shape[3], :model_dict[k].shape[4]]
                    else:
                        raise Exception
            else:
                try:
                    if 'adaptor1_base' in k:
                        if model_dict[k].shape == state_dict['layer1.2'+k[13:]].shape:
                            model_dict[k] = state_dict['layer1.2'+k[13:]]
                            # print('Done, adaptor', k)
                        else:
                            print('Skip, adaptor', k)
                    elif 'adaptor1_sub' in k:
                        if 'conv3' in k:
                            v = state_dict['layer1.2'+k[12:]]
                            Cout, Cin, H, W = v.shape
                            temp = ops.avg_pool1d(ops.Transpose()(mindspore.Tensor(v), (0, 2, 3, 1)).reshape(Cout, H*W, Cin), kernel_size=K, stride=K).reshape(Cout, H, W, -1)
                            model_dict[k] = ops.Transpose()(mindspore.Tensor(temp), (0, 3, 1, 2)).tile((K, 1, 1, 1))
                            # model_dict[k] = F.avg_pool1d(v.permute(0, 2, 3, 1).reshape(Cout, H*W, Cin), kernel_size=K).reshape(Cout, H, W, -1).permute(0, 3, 1, 2).repeat(K, 1, 1, 1)
                        elif 'bn3' in k:
                            v = state_dict['layer1.2'+k[12:]]
                            model_dict[k] = v.repeat(K)
                        elif model_dict[k].shape == state_dict['layer1.2'+k[12:]].shape:
                            model_dict[k] = state_dict['layer1.2'+k[12:]]
                        else:
                            v = state_dict['layer1.2'+k[12:]]
                            Cout, Cin, H, W = v.shape
                            temp = ops.avg_pool1d(ops.Transpose()(mindspore.Tensor(v), (0, 2, 3, 1)).reshape(Cout, H*W, Cin), kernel_size=K, stride=K).reshape(Cout, H, W, -1)
                            model_dict[k] = ops.Transpose()(mindspore.Tensor(temp), (0, 3, 1, 2))
                            # model_dict[k] = F.avg_pool1d(v.permute(0, 2, 3, 1).reshape(Cout, H*W, Cin), kernel_size=K).reshape(Cout, H, W, -1).permute(0, 3, 1, 2)
                        # print('Done, adaptor', k)
                    elif 'adaptor2_base' in k:
                        if model_dict[k].shape == state_dict['layer2.3'+k[13:]].shape:
                            model_dict[k] = state_dict['layer2.3'+k[13:]]
                            # print('Done, adaptor', k)
                        else:
                            print('Skip, adaptor', k)
                    elif 'adaptor2_sub' in k:
                        if 'conv3' in k:
                            v = state_dict['layer2.3'+k[12:]]
                            Cout, Cin, H, W = v.shape
                            temp = ops.avg_pool1d(ops.Transpose()(mindspore.Tensor(v), (0, 2, 3, 1)).reshape(Cout, H*W, Cin), kernel_size=K, stride=K).reshape(Cout, H, W, -1)
                            model_dict[k] = ops.Transpose()(mindspore.Tensor(temp), (0, 3, 1, 2)).tile((K, 1, 1, 1))
                            # model_dict[k] = F.avg_pool1d(v.permute(0, 2, 3, 1).reshape(Cout, H*W, Cin), kernel_size=K).reshape(Cout, H, W, -1).permute(0, 3, 1, 2).repeat(K, 1, 1, 1)
                        elif 'bn3' in k:
                            v = state_dict['layer2.3'+k[12:]]
                            model_dict[k] = v.repeat(K)
                        elif model_dict[k].shape == state_dict['layer2.3'+k[12:]].shape:
                            model_dict[k] = state_dict['layer2.3'+k[12:]]
                        else:
                            v = state_dict['layer2.3'+k[12:]]
                            Cout, Cin, H, W = v.shape
                            temp = ops.avg_pool1d(ops.Transpose()(mindspore.Tensor(v), (0, 2, 3, 1)).reshape(Cout, H*W, Cin), kernel_size=K, stride=K).reshape(Cout, H, W, -1)
                            model_dict[k] = ops.Transpose()(mindspore.Tensor(temp), (0, 3, 1, 2))
                            # model_dict[k] = F.avg_pool1d(v.permute(0, 2, 3, 1).reshape(Cout, H*W, Cin), kernel_size=K).reshape(Cout, H, W, -1).permute(0, 3, 1, 2)
                        # print('Done, adaptor', k)

                    elif 'adaptor3_base' in k:
                        if model_dict[k].shape == state_dict['layer3.5'+k[13:]].shape:
                            model_dict[k] = state_dict['layer3.5'+k[13:]]
                            # print('Done, adaptor', k)
                        else:
                            print('Skip, adaptor', k)
                    elif 'adaptor3_sub' in k:
                        if 'conv3' in k:
                            v = state_dict['layer3.5'+k[12:]]
                            Cout, Cin, H, W = v.shape
                            temp = ops.avg_pool1d(ops.Transpose()(mindspore.Tensor(v), (0, 2, 3, 1)).reshape(Cout, H*W, Cin), kernel_size=K, stride=K).reshape(Cout, H, W, -1)
                            model_dict[k] = ops.Transpose()(mindspore.Tensor(temp), (0, 3, 1, 2)).tile((K, 1, 1, 1))
                            # model_dict[k] = F.avg_pool1d(v.permute(0, 2, 3, 1).reshape(Cout, H*W, Cin), kernel_size=K).reshape(Cout, H, W, -1).permute(0, 3, 1, 2).repeat(K, 1, 1, 1)
                        elif 'bn3' in k:
                            v = state_dict['layer3.5'+k[12:]]
                            model_dict[k] = v.repeat(K)
                        elif model_dict[k].shape == state_dict['layer3.5'+k[12:]].shape:
                            model_dict[k] = state_dict['layer3.5'+k[12:]]
                        else:
                            v = state_dict['layer3.5'+k[12:]]
                            Cout, Cin, H, W = v.shape
                            temp = ops.avg_pool1d(ops.Transpose()(mindspore.Tensor(v), (0, 2, 3, 1)).reshape(Cout, H*W, Cin), kernel_size=K, stride=K).reshape(Cout, H, W, -1)
                            model_dict[k] = ops.Transpose()(mindspore.Tensor(temp), (0, 3, 1, 2))
                            # model_dict[k] = F.avg_pool1d(v.permute(0, 2, 3, 1).reshape(Cout, H*W, Cin), kernel_size=K).reshape(Cout, H, W, -1).permute(0, 3, 1, 2)
                        # print('Done, adaptor', k)

                    elif 'adaptor4_base' in k:
                        if model_dict[k].shape == state_dict['layer4.2'+k[13:]].shape:
                            model_dict[k] = state_dict['layer4.2'+k[13:]]
                            # print('Done, adaptor', k)
                        else:
                            print('Skip, adaptor', k)
                    elif 'adaptor4_sub' in k:
                        if 'conv3' in k:
                            v = state_dict['layer4.2'+k[12:]]
                            Cout, Cin, H, W = v.shape
                            temp = ops.avg_pool1d(ops.Transpose()(mindspore.Tensor(v), (0, 2, 3, 1)).reshape(Cout, H*W, Cin), kernel_size=K, stride=K).reshape(Cout, H, W, -1)
                            model_dict[k] = ops.Transpose()(mindspore.Tensor(temp), (0, 3, 1, 2)).tile((K, 1, 1, 1))
                            # model_dict[k] = F.avg_pool1d(v.permute(0, 2, 3, 1).reshape(Cout, H*W, Cin), kernel_size=K).reshape(Cout, H, W, -1).permute(0, 3, 1, 2).repeat(K, 1, 1, 1)
                        elif 'bn3' in k:
                            v = state_dict['layer4.2'+k[12:]]
                            model_dict[k] = v.repeat(K)
                        elif 'bn1' in k:
                            if 'IN' in k:
                                model_dict[k] = state_dict['layer4.2.bn1.'+k.split('.')[-1]][:256]
                            else:
                                model_dict[k] = state_dict['layer4.2.bn1.'+k.split('.')[-1]][256:]
                        elif model_dict[k].shape == state_dict['layer4.2'+k[12:]].shape:
                            model_dict[k] = state_dict['layer4.2'+k[12:]]
                        else:
                            v = state_dict['layer4.2'+k[12:]]
                            Cout, Cin, H, W = v.shape
                            temp = ops.avg_pool1d(ops.Transpose()(mindspore.Tensor(v), (0, 2, 3, 1)).reshape(Cout, H*W, Cin), kernel_size=K, stride=K).reshape(Cout, H, W, -1)
                            model_dict[k] = ops.Transpose()(mindspore.Tensor(temp), (0, 3, 1, 2))
                            # model_dict[k] = F.avg_pool1d(v.permute(0, 2, 3, 1).reshape(Cout, H*W, Cin), kernel_size=K).reshape(Cout, H, W, -1).permute(0, 3, 1, 2)
                        # print('Done, adaptor', k)

                except Exception:
                    pass

        mindspore.save_checkpoint([{"name": key, "data": mindspore.Tensor(value.numpy())} for key, value in model_dict.items()], "/home/yuming/.cache/torch/checkpoints/ACL-DGReID.ckpt")
        # print(2)
        # print([{"name": key, "data": mindspore.Tensor(value.numpy())} for key, value in model_dict.items()][0])
        incompatible = mindspore.load_checkpoint("/home/yuming/.cache/torch/checkpoints/ACL-DGReID.ckpt", model)
        # incompatible = model.load_state_dict(model_dict, strict=False)
        # if incompatible.missing_keys:
        #     logger.info(
        #         get_missing_parameters_message(incompatible.missing_keys)
        #     )
        # if incompatible.unexpected_keys:
        #     logger.info(
        #         get_unexpected_parameters_message(incompatible.unexpected_keys)
        #     )

    return model