# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# import torch
# import torch.nn.functional as F
# from torch import nn
from mindspore import nn
import mindspore

from collections import OrderedDict

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
# from fastreid.modeling.ops import MetaConv2d, MetaBNNorm, MetaParam
from fastreid.modeling.ops import MetaLinear, MetaConv2d, MetaBNNorm, MetaParam
from .build import REID_HEADS_REGISTRY


# class Sequential_ext(nn.Module):
#     """A Sequential container extended to also propagate the gating information
#     that is needed in the target rate loss.
#     """

#     def __init__(self, *args):
#         super(Sequential_ext, self).__init__()
#         if len(args) == 1 and isinstance(args[0], OrderedDict):
#             for key, module in args[0].items():
#                 self.add_module(key, module)
#         else:
#             for idx, module in enumerate(args):
#                 self.add_module(str(idx), module)

#     def __getitem__(self, idx):
#         if not (-len(self) <= idx < len(self)):
#             raise IndexError('index {} is out of range'.format(idx))
#         if idx < 0:
#             idx += len(self)
#         it = iter(self._modules.values())
#         for i in range(idx):
#             next(it)
#         return next(it)

#     def __len__(self):
#         return len(self._modules)

#     def forward(self, input, opt):
#         for i, module in enumerate(self._modules.values()):
#             input = module(input, opt)
#         return input

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


@REID_HEADS_REGISTRY.register()
# class MetaEmbeddingHead(nn.Module):
class MetaEmbeddingHead(nn.Cell):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes1,
            num_classes2,
            num_classes3,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat

        neck = []
        if embedding_dim > 0:
            neck.append(MetaConv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(MetaBNNorm(feat_dim, bias_freeze=True))

        self.bottleneck = Sequential_ext(*neck)

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        self.weight1 = MetaLinear(feat_dim, num_classes1)
        self.weight2 = MetaLinear(feat_dim, num_classes2)
        self.weight3 = MetaLinear(feat_dim, num_classes3)
        self.center = MetaParam(feat_dim, num_classes1+num_classes2+num_classes3)


        self.cls_layer1 = getattr(any_softmax, cls_type)(num_classes1, scale, margin)
        self.cls_layer2 = getattr(any_softmax, cls_type)(num_classes2, scale, margin)
        self.cls_layer3 = getattr(any_softmax, cls_type)(num_classes3, scale, margin)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)
        # print("self.bottleneck.gamma", self.bottleneck[0].gamma.value())
        # print("self.bottleneck.beta", self.bottleneck[0].beta.value())
        self.weight1.weight.data.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.Normal(sigma=0.01, mean=0.0), self.weight1.weight.data.shape, self.weight1.weight.data.dtype))
        self.weight2.weight.data.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.Normal(sigma=0.01, mean=0.0), self.weight2.weight.data.shape, self.weight2.weight.data.dtype))
        self.weight3.weight.data.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.Normal(sigma=0.01, mean=0.0), self.weight3.weight.data.shape, self.weight3.weight.data.dtype))
        self.center.centers.data.set_data(mindspore.common.initializer.initializer(mindspore.common.initializer.Normal(sigma=0.01, mean=0.0), self.center.centers.data.shape, self.center.centers.data.dtype))
        # nn.init.normal_(self.weight1.weight.data, std=0.01)
        # nn.init.normal_(self.weight2.weight.data, std=0.01)
        # nn.init.normal_(self.weight3.weight.data, std=0.01)
        # nn.init.normal_(self.center.centers.data, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes1   = cfg.MODEL.HEADS.NUM_CLASSES1
        num_classes2   = cfg.MODEL.HEADS.NUM_CLASSES2
        num_classes3   = cfg.MODEL.HEADS.NUM_CLASSES3
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes1': num_classes1,
            'num_classes2': num_classes2,
            'num_classes3': num_classes3,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    #CHANGE Add reduction version
    def construct(self, features, targets=None, opt=-1):
    # def forward(self, features, targets=None, opt=-1):
    # def forward(self, features, targets=None, opt=-1):
        """
        See :class:`ReIDHeads.forward`.
        """
        # print("self.pool_layer", type(self.pool_layer), self.pool_layer)
        pool_feat = self.pool_layer(features)

        # if opt['meta']:
        #     import pdb; pdb.set_trace()

        # print("ms pool_feat", type(pool_feat), pool_feat)
        neck_feat = self.bottleneck(pool_feat, opt)

        # print("ms neck_feat", type(neck_feat), neck_feat)
        neck_feat = neck_feat[..., 0, 0]
        # print("neck_feat", type(neck_feat), neck_feat)

        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on

        # Training
        if self.cls_layer1.__class__.__name__ == 'Linear':
            logits1 = self.weight1(neck_feat, opt)
            logits2 = self.weight2(neck_feat, opt)
            logits3 = self.weight3(neck_feat, opt)

            center_distmat = self.center(neck_feat, opt)

        else:
            # l2_normalize = mindspore.ops.L2Normalize(axis=0)
            neck_feat_normalized = mindspore.ops.L2Normalize(axis=0)(neck_feat)
            weight1_normalized = mindspore.ops.L2Normalize(axis=0)(self.weight1)
            weight2_normalized = mindspore.ops.L2Normalize(axis=0)(self.weight2)
            weight3_normalized = mindspore.ops.L2Normalize(axis=0)(self.weight3)
            logits1 = mindspore.ops.dense(neck_feat_normalized, weight1_normalized)
            logits2 = mindspore.ops.dense(neck_feat_normalized, weight2_normalized)
            logits3 = mindspore.ops.dense(neck_feat_normalized, weight3_normalized)
            # logits1 = F.linear(F.normalize(neck_feat), F.normalize(self.weight1))
            # logits2 = F.linear(F.normalize(neck_feat), F.normalize(self.weight2))
            # logits3 = F.linear(F.normalize(neck_feat), F.normalize(self.weight3))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs1 = self.cls_layer1(logits1, targets)
        cls_outputs2 = self.cls_layer2(logits2, targets)
        cls_outputs3 = self.cls_layer3(logits3, targets)
        # cls_outputs1 = self.cls_layer1(logits1.clone(), targets)
        # cls_outputs2 = self.cls_layer2(logits2.clone(), targets)
        # cls_outputs3 = self.cls_layer3(logits3.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before':  feat = pool_feat[..., 0, 0]
        elif self.neck_feat == 'after': feat = mindspore.ops.L2Normalize(axis=1)(neck_feat)
        # elif self.neck_feat == 'after': feat = F.normalize(neck_feat, 2, 1)
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on
        # print("logits1", type(logits1), logits1)
        # print("self.cls_layer1.s", type(self.cls_layer1.s), self.cls_layer1.s)
        # print("logits1.mul(self.cls_layer1.s)", type(logits1.mul(self.cls_layer1.s)), logits1.mul(self.cls_layer1.s))
        return cls_outputs1, cls_outputs2, cls_outputs3, center_distmat, logits1.mul(self.cls_layer1.s), logits2.mul(self.cls_layer2.s), logits3.mul(self.cls_layer3.s), feat
        # return {
        #     "cls_outputs1": cls_outputs1,
        #     "cls_outputs2": cls_outputs2,
        #     "cls_outputs3": cls_outputs3,
        #     "center_distmat": center_distmat,
        #     "pred_class_logits1": logits1.mul(self.cls_layer1.s),
        #     "pred_class_logits2": logits2.mul(self.cls_layer2.s),
        #     "pred_class_logits3": logits3.mul(self.cls_layer3.s),
        #     "features": feat,
        # }
