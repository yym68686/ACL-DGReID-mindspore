# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
from os import path
import numpy as np
from fastreid.modeling.losses.cluster_loss import intraCluster, interCluster
from fastreid.modeling.losses.center_loss import centerLoss
from fastreid.modeling.losses.triplet_loss import triplet_loss
from fastreid.modeling.losses.triplet_loss_MetaIBN import triplet_loss_Meta
from fastreid.modeling.losses.domain_SCT_loss import domain_SCT_loss
# import torch
# from torch import nn
from mindspore import nn
import mindspore
# import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


def process_state_dict(state_dict):
    result_dict = {}
    for k in state_dict:
        if 'backbone' in k:
            result_dict[k.split('backbone.')[1]] = state_dict[k]

    return result_dict


@META_ARCH_REGISTRY.register()
# class Baseline(nn.Module):
class Baseline(nn.Cell):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            # router,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone
        # head
        self.heads = heads

        self.loss_kwargs = loss_kwargs

        # print("pixel_mean", type(pixel_mean), pixel_mean)
        self.pixel_mean = mindspore.Tensor(pixel_mean).view(1, -1, 1, 1)
        # self.pixel_mean = mindspore.Tensor(self.pixel_mean)
        # print("self.pixel_mean", type(self.pixel_mean), self.pixel_mean)
        self.pixel_std = mindspore.Tensor(pixel_std).view(1, -1, 1, 1)
        # self.pixel_std = mindspore.Tensor(self.pixel_std)
        # self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        # self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)
        # self.conv1 = nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1, pad_mode="pad")
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1, pad_mode="pad")
        # self.pool2 = nn.MaxPool2d(kernel_size=8, stride=8)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)

        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    # def forward(self, batched_inputs, epoch, opt=-1):
    # def construct(self, images, targets, camids, domainids, img_paths, epoch, opt=-1):
    # def construct(self, images, targets, domainids, epoch, opt=-1):
    def construct(self, images, epoch, opt=-1):
        # print("type(batched_inputs['images'])", images)
        images = self.preprocess_image(images)
        # print("images", type(images), images)

        # print("images.shape", images.shape)
        # images (8, 3, 256, 128) -> features (8, 2048, 16, 8)
        # features = self.conv1(images)
        # features = self.pool1(features)
        # features = self.conv2(features)
        # features = self.pool2(features)
        # # paths (8, 16)
        # paths = mindspore.Tensor(np.random.randn(8, 16), mindspore.float32)
        features, paths, _ = self.backbone(images, epoch, opt)
        # print("features.shape", features.shape)
        # print("paths.shape", paths.shape)
        # features = mindspore.Tensor(features)
        # print("features", type(features), features)
        # print("targets" in batched_inputs)

        if self.training:
            assert targets is not None, "Person ID annotation are missing in training!"
            # assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            # print("batched_inputs", len(batched_inputs), type(batched_inputs), batched_inputs)
            # targets = batched_inputs["targets"]

            # print("targets.sum()", targets.sum())
            domain_ids = domainids
            # domain_ids = batched_inputs["domainids"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            # if targets.sum() < 0: targets.zero_()

            # print("targets", targets)
            pred_class_logits1, pred_class_logits2, pred_class_logits3, center_distmat, cls_outputs1, cls_outputs2, cls_outputs3, pred_features = self.heads(features, targets, opt=opt)
            # outputs = self.heads(features, targets, opt=opt)
            # for key, value in outputs.items():
            #     print(f"{key}", value.shape)
            # pred_class_logits1 = outputs['pred_class_logits1']
            # pred_class_logits2 = outputs['pred_class_logits2']
            # pred_class_logits3 = outputs['pred_class_logits3']
            # # pred_class_logits1 = outputs['pred_class_logits1'].detach()
            # # pred_class_logits2 = outputs['pred_class_logits2'].detach()
            # # pred_class_logits3 = outputs['pred_class_logits3'].detach()
            # cls_outputs1       = outputs['cls_outputs1']
            # cls_outputs2       = outputs['cls_outputs2']
            # cls_outputs3       = outputs['cls_outputs3']
            # pred_features     = outputs['features']
            # print("outputs", type(outputs), outputs)
            loss_cls, loss_center, loss_triplet, loss_circle, loss_cosface, loss_triplet_add, loss_triplet_mtrain, loss_stc, loss_triplet_mtest = self.losses(pred_class_logits1, pred_class_logits2, pred_class_logits3, cls_outputs1, cls_outputs2, cls_outputs3, pred_features, targets, domain_ids, paths, opt)
            # losses = self.losses(pred_class_logits1, pred_class_logits2, pred_class_logits3, cls_outputs1, cls_outputs2, cls_outputs3, pred_features, targets, domain_ids, paths, opt)
            # print("targets", type(targets), targets)

            loss_domain_intra = intraCluster(paths, domain_ids)
            # losses['loss_domain_intra'] = intraCluster(paths, domain_ids)
            # print("targets loss_Center 1", type(targets), targets)
            # QUES
            loss_domain_inter = interCluster(paths, domain_ids)
            # losses['loss_domain_inter'] = interCluster(paths, domain_ids)


            # print("targets loss_Center 2", type(targets), targets)
            if opt == -1 or opt['type'] == 'basic':
                # pass
                # print("targets loss_Center", type(targets), targets)
                loss_Center_ = centerLoss(center_distmat, targets) * 5e-4
                # losses['loss_Center'] = centerLoss(center_distmat, targets) * 5e-4
                # losses['loss_Center'] = centerLoss(outputs['center_distmat'], targets) * 5e-4

            elif opt['type'] == 'mtrain':
                pass

            elif opt['type'] == 'mtest':
                loss_Center_ = centerLoss(center_distmat, targets) * 1e-3
                # losses['loss_Center'] = centerLoss(center_distmat, targets) * 1e-3
            else:
                raise NotImplementedError

            # print("domain_ids", type(domain_ids), domain_ids.shape, domain_ids)
            return loss_cls, loss_center, loss_triplet, loss_circle, loss_cosface, loss_triplet_add, loss_triplet_mtrain, loss_stc, loss_triplet_mtest, loss_domain_intra, loss_domain_inter, loss_Center_
            # return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        images = 0
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
            # print(222)
        # elif isinstance(batched_inputs, torch.Tensor):
        elif isinstance(batched_inputs, mindspore.Tensor):
            images = batched_inputs
            # print("batched_inputs", batched_inputs)
        else:
            # images = mindspore.Tensor(batched_inputs)
            if isinstance(batched_inputs, list) or isinstance(batched_inputs, tuple):
                # print("batched_inputs", batched_inputs)
                images = batched_inputs[0]
                # tmp_images = batched_inputs[0]["images"]
                # tmp_images = batched_inputs
                # tmp_images = [item["images"].tolist() for item in batched_inputs]
                # tmp_images = mindspore.Tensor(np.array(tmp_images).astype(np.float32), mindspore.float32)
                # tmp_images = mindspore.Tensor(tmp_images.numpy().astype(np.float32), mindspore.float32)
                # print(type(tmp_images))
                # if isinstance(tmp_images, mindspore.Tensor):
                #     images = tmp_images
                # print(111)
            # raise TypeError("batched_inputs must be dict or mindspore.Tensor, but get {}".format(type(batched_inputs)))
            # raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))


        # images = mindspore.Tensor(np.array(images).astype(np.float32), mindspore.float32)
        # print("images", images)
        # print("self.pixel_mean", len(self.pixel_mean), type(self.pixel_mean), self.pixel_mean)
        images = images.sub(self.pixel_mean).div(self.pixel_std)
        # images = mindspore.Tensor(images)
        # print("images", len(images), type(images), images)
        # images = mindspore.ops.sub(images, self.pixel_mean)
        # images = mindspore.ops.div(images, self.pixel_std)
        # images.sub(self.pixel_mean).div(self.pixel_std)
        # images.sub(self.pixel_mean).div(self.pixel_std)
        # images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, pred_class_logits1, pred_class_logits2, pred_class_logits3, cls_outputs1, cls_outputs2, cls_outputs3, pred_features, gt_labels, domain_labels=None, paths=None, opt=-1):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        # print("outputs", type(outputs), outputs)
        # pred_class_logits1 = outputs['pred_class_logits1']
        # pred_class_logits2 = outputs['pred_class_logits2']
        # pred_class_logits3 = outputs['pred_class_logits3']
        # # pred_class_logits1 = outputs['pred_class_logits1'].detach()
        # # pred_class_logits2 = outputs['pred_class_logits2'].detach()
        # # pred_class_logits3 = outputs['pred_class_logits3'].detach()
        # cls_outputs1       = outputs['cls_outputs1']
        # cls_outputs2       = outputs['cls_outputs2']
        # cls_outputs3       = outputs['cls_outputs3']
        # pred_features     = outputs['features']
        # fmt: on

        # print("cls_outputs1.shape", cls_outputs1.shape)
        num_classes1 = cls_outputs1.shape[-1]
        num_classes2 = cls_outputs1.shape[-1] + cls_outputs2.shape[-1]
        num_classes3 = cls_outputs1.shape[-1] + cls_outputs2.shape[-1] + cls_outputs3.shape[-1]
        # print("num_classes1", type(num_classes1), num_classes1)
        # print("gt_labels", type(gt_labels), gt_labels)
        # idx1 = int(gt_labels) < num_classes1
        # idx2 = (int(gt_labels) < num_classes2) & (int(gt_labels) >= num_classes1)
        # idx3 = (int(gt_labels) < num_classes3) & (int(gt_labels) >= num_classes2)
        idx1 = gt_labels < num_classes1
        idx2 = (gt_labels < num_classes2) & (gt_labels >= num_classes1)
        idx3 = (gt_labels < num_classes3) & (gt_labels >= num_classes2)

        # Log prediction accuracy
        # print("pred_class_logits1", type(pred_class_logits1), pred_class_logits1)
        # if len(gt_labels.shape) > 1:
        #     print("len(gt_labels) > 1!!!!!!!!!!!!!!!!!!")
        #     exit(0)
        # if idx1:
        #     # print("gt_labels[idx1]", type(gt_labels[idx1]), gt_labels)
        #     log_accuracy(pred_class_logits1[idx1], gt_labels)
        # if idx2:
        #     log_accuracy(pred_class_logits2[idx2], gt_labels)
        # if idx3:
        #     # print("gt_labels[idx3]", type(gt_labels[idx3]), gt_labels)
        #     log_accuracy(pred_class_logits3[idx3], gt_labels)
        if idx1.sum().item():
            log_accuracy(pred_class_logits1[idx1], gt_labels[idx1])
        if idx2.sum().item():
            log_accuracy(pred_class_logits2[idx2], gt_labels[idx2])
        if idx3.sum().item():
            log_accuracy(pred_class_logits3[idx3], gt_labels[idx3])

        # loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        loss_triplet_add = 0
        loss_triplet_mtrain = 0
        loss_stc = 0
        loss_triplet_mtest = 0
        if opt == -1 or opt['type'] == 'basic':

            if 'CrossEntropyLoss' in loss_names:
                ce_kwargs = self.loss_kwargs.get('ce')
                count = 0
                temp_loss = 0
                for i in range(1, 4):
                    if eval('idx'+str(i)).sum().item() == 0:
                    # if eval('idx'+str(i)) == 0:
                        continue
                    count += 1
                    # temp_loss += 1
                    # QUES
                    # print("eval('cls_outputs'+str(i))[eval('idx'+str(i))]", eval('cls_outputs'+str(i)).shape, 'idx'+str(i), eval('idx'+str(i)), eval('cls_outputs'+str(i))[eval('idx'+str(i))].shape, eval('cls_outputs'+str(i))[eval('idx'+str(i))])
                    # print("gt_labels[eval('idx'+str(i))]-eval('num_classes'+str(i-1))", gt_labels[eval('idx'+str(i))]-eval('num_classes'+str(i-1)))
                    # print("gt_labels", type(gt_labels), gt_labels.shape, gt_labels)
                    # kkk = mindspore.Tensor(gt_labels-eval('num_classes'+str(i-1))) if i > 1 else gt_labels
                    # print("kkk", type(kkk), kkk, gt_labels, eval('num_classes'+str(i-1)) if i > 1 else 0, mindspore.Tensor(gt_labels-eval('num_classes'+str(i-1))) if i > 1 else 0)
                    # kkk = gt_labels[eval('idx'+str(i))]-eval('num_classes'+str(i-1)) if i > 1 else gt_labels[eval('idx'+str(i))]
                    # print("kkk", type(kkk), kkk, gt_labels[eval('idx'+str(i))], (gt_labels[eval('idx'+str(i))]-eval('num_classes'+str(i-1))) if i > 1 else 0)
                    # if len(gt_labels.shape) == 0:
                    #     temp_loss += cross_entropy_loss(
                    #         eval('cls_outputs'+str(i))[eval('idx'+str(i))],
                    #         mindspore.Tensor(gt_labels - eval('num_classes'+str(i-1))) if i > 1 else gt_labels,
                    #         # mindspore.Tensor(gt_labels[eval('idx'+str(i))] - eval('num_classes'+str(i-1))) if i > 1 else mindspore.Tensor(gt_labels[eval('idx'+str(i))]),
                    #         ce_kwargs.get('eps'),
                    #         ce_kwargs.get('alpha')
                    #     ) * ce_kwargs.get('scale')
                    temp_loss += cross_entropy_loss(
                        eval('cls_outputs'+str(i))[eval('idx'+str(i))],
                        gt_labels[eval('idx'+str(i))]-eval('num_classes'+str(i-1)) if i > 1 else gt_labels[eval('idx'+str(i))],
                        ce_kwargs.get('eps'),
                        ce_kwargs.get('alpha')
                    ) * ce_kwargs.get('scale')
                loss_cls = temp_loss / count
                # loss_dict['loss_cls'] = temp_loss / count

            loss_center = 0
            if 'CenterLoss' in loss_names:
                loss_center = 5e-4 * self.center_loss(
                    pred_features,
                    gt_labels
                )
                # loss_dict['loss_center'] = 5e-4 * self.center_loss(
                #     pred_features,
                #     gt_labels
                # )
            loss_triplet = 0
            if 'TripletLoss' in loss_names:
                tri_kwargs = self.loss_kwargs.get('tri')
                loss_triplet = triplet_loss(
                    pred_features,
                    gt_labels,
                    tri_kwargs.get('margin'),
                    tri_kwargs.get('norm_feat'),
                    tri_kwargs.get('hard_mining')
                ) * tri_kwargs.get('scale')
                # loss_dict['loss_triplet'] = triplet_loss(
                #     pred_features,
                #     gt_labels,
                #     tri_kwargs.get('margin'),
                #     tri_kwargs.get('norm_feat'),
                #     tri_kwargs.get('hard_mining')
                # ) * tri_kwargs.get('scale')

            loss_circle = 0
            if 'CircleLoss' in loss_names:
                circle_kwargs = self.loss_kwargs.get('circle')
                loss_circle = pairwise_circleloss(
                    pred_features,
                    gt_labels,
                    circle_kwargs.get('margin'),
                    circle_kwargs.get('gamma')
                ) * circle_kwargs.get('scale')
                # loss_dict['loss_circle'] = pairwise_circleloss(
                #     pred_features,
                #     gt_labels,
                #     circle_kwargs.get('margin'),
                #     circle_kwargs.get('gamma')
                # ) * circle_kwargs.get('scale')

            loss_cosface = 0
            if 'Cosface' in loss_names:
                cosface_kwargs = self.loss_kwargs.get('cosface')
                loss_cosface = pairwise_cosface(
                    pred_features,
                    gt_labels,
                    cosface_kwargs.get('margin'),
                    cosface_kwargs.get('gamma'),
                ) * cosface_kwargs.get('scale')
                # loss_dict['loss_cosface'] = pairwise_cosface(
                #     pred_features,
                #     gt_labels,
                #     cosface_kwargs.get('margin'),
                #     cosface_kwargs.get('gamma'),
                # ) * cosface_kwargs.get('scale')

        elif opt['type'] == 'mtrain':

            loss_triplet_add = triplet_loss_Meta(
                pred_features,
                gt_labels,
                0.0,
                False,
                True,
                'euclidean',
                'logistic',
                domain_labels,
                [0, 0, 1],
                [0, 1, 0],
            )
            # loss_dict['loss_triplet_add'] = triplet_loss_Meta(
            #     pred_features,
            #     gt_labels,
            #     0.0,
            #     False,
            #     True,
            #     'euclidean',
            #     'logistic',
            #     domain_labels,
            #     [0, 0, 1],
            #     [0, 1, 0],
            # )

            loss_triplet_mtrain = triplet_loss_Meta(
                pred_features,
                gt_labels,
                0.3,
                False,
                True,
                'euclidean',
                'logistic',
                domain_labels,
                [1, 0, 0],
                [0, 1, 1],
            )
            # loss_dict['loss_triplet_mtrain'] = triplet_loss_Meta(
            #     pred_features,
            #     gt_labels,
            #     0.3,
            #     False,
            #     True,
            #     'euclidean',
            #     'logistic',
            #     domain_labels,
            #     [1, 0, 0],
            #     [0, 1, 1],
            # )

            loss_stc = domain_SCT_loss(
                pred_features,
                domain_labels,
                True,
                'cosine_sim',
            )
            # loss_dict['loss_stc'] = domain_SCT_loss(
            #     pred_features,
            #     domain_labels,
            #     True,
            #     'cosine_sim',
            # )

        elif opt['type'] == 'mtest':

            loss_triplet_mtest = triplet_loss_Meta(
                pred_features,
                gt_labels,
                0.3,
                False,
                True,
                'euclidean',
                'logistic',
                domain_labels,
                [1, 0, 0],
                [0, 1, 1],
            )
            # loss_dict['loss_triplet_mtest'] = triplet_loss_Meta(
            #     pred_features,
            #     gt_labels,
            #     0.3,
            #     False,
            #     True,
            #     'euclidean',
            #     'logistic',
            #     domain_labels,
            #     [1, 0, 0],
            #     [0, 1, 1],
            # )

        else:
            raise NotImplementedError


        return loss_cls, loss_center, loss_triplet, loss_circle, loss_cosface, loss_triplet_add, loss_triplet_mtrain, loss_stc, loss_triplet_mtest
        # return loss_dict
