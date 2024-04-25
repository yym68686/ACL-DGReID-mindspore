# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# import os
# os.system("clear")
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import copy
from os import path
from pytorch_fastreid.modeling.losses.cluster_loss import intraCluster, interCluster
from pytorch_fastreid.modeling.losses.center_loss import centerLoss
from pytorch_fastreid.modeling.losses.triplet_loss import triplet_loss
from pytorch_fastreid.modeling.losses.triplet_loss_MetaIBN import triplet_loss_Meta
from pytorch_fastreid.modeling.losses.domain_SCT_loss import domain_SCT_loss
import torch
from torch import nn
import torch.nn.functional as F

from pytorch_fastreid.config import configurable
from pytorch_fastreid.modeling.backbones import build_backbone
from pytorch_fastreid.modeling.heads import build_heads
from pytorch_fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


def process_state_dict(state_dict):
    result_dict = {}
    for k in state_dict:
        if 'backbone' in k:
            result_dict[k.split('backbone.')[1]] = state_dict[k]

    return result_dict


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
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

        # print("__init__ heads.weight before", heads.bottleneck[0].weight)
        self.heads = heads
        # print("__init__ self.heads.weight after", self.heads.bottleneck[0].weight)

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)
        # print("__init__ self.heads.weight last", self.heads.bottleneck[0].weight)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        # print("from_config heads.weight before", heads.bottleneck[0].weight)

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

    def forward(self, batched_inputs, epoch, opt=None):
        # print("self.heads.weight before", self.heads.bottleneck[0].weight)
        images = self.preprocess_image(batched_inputs)

        features, paths, _ = self.backbone(images, epoch, opt)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]
            domain_ids = batched_inputs["domainids"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets, opt=opt)
            losses = self.losses(outputs, targets, domain_ids, paths, opt)

            losses['loss_domain_intra'] = intraCluster(paths, domain_ids)
            losses['loss_domain_inter'] = interCluster(paths, domain_ids)


            if opt == None or opt['type'] == 'basic':
                # pass
                losses['loss_Center'] = centerLoss(outputs['center_distmat'], targets) * 5e-4

            elif opt['type'] == 'mtrain':
                pass

            elif opt['type'] == 'mtest':
                losses['loss_Center'] = centerLoss(outputs['center_distmat'], targets) * 1e-3
            else:
                raise NotImplementedError

            return losses
        else:
            # print("self.heads.weight", self.heads.bottleneck[0].weight)
            # print("self.heads.weight", self.heads[1][0].weight)
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels, domain_labels=None, paths=None, opt=None):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits1 = outputs['pred_class_logits1'].detach()
        pred_class_logits2 = outputs['pred_class_logits2'].detach()
        pred_class_logits3 = outputs['pred_class_logits3'].detach()
        cls_outputs1       = outputs['cls_outputs1']
        cls_outputs2       = outputs['cls_outputs2']
        cls_outputs3       = outputs['cls_outputs3']
        pred_features     = outputs['features']
        # fmt: on

        num_classes1 = cls_outputs1.shape[-1]
        num_classes2 = cls_outputs1.shape[-1] + cls_outputs2.shape[-1]
        num_classes3 = cls_outputs1.shape[-1] + cls_outputs2.shape[-1] + cls_outputs3.shape[-1]
        idx1 = gt_labels < num_classes1
        idx2 = (gt_labels < num_classes2) & (gt_labels >= num_classes1)
        idx3 = (gt_labels < num_classes3) & (gt_labels >= num_classes2)
        # Log prediction accuracy
        if idx1.sum().item():
            log_accuracy(pred_class_logits1[idx1], gt_labels[idx1])
        if idx2.sum().item():
            log_accuracy(pred_class_logits2[idx2], gt_labels[idx2])
        if idx3.sum().item():
            log_accuracy(pred_class_logits3[idx3], gt_labels[idx3])

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']


        if opt == None or opt['type'] == 'basic':

            if 'CrossEntropyLoss' in loss_names:
                ce_kwargs = self.loss_kwargs.get('ce')
                count = 0
                temp_loss = 0
                for i in range(1, 4):
                    if eval('idx'+str(i)).sum().item() == 0:
                        continue
                    count += 1
                    temp_loss += cross_entropy_loss(
                        eval('cls_outputs'+str(i))[eval('idx'+str(i))],
                        gt_labels[eval('idx'+str(i))]-eval('num_classes'+str(i-1)) if i > 1 else gt_labels[eval('idx'+str(i))],
                        ce_kwargs.get('eps'),
                        ce_kwargs.get('alpha')
                    ) * ce_kwargs.get('scale')
                loss_dict['loss_cls'] = temp_loss / count

            if 'CenterLoss' in loss_names:
                loss_dict['loss_center'] = 5e-4 * self.center_loss(
                    pred_features,
                    gt_labels
                )

            if 'TripletLoss' in loss_names:
                tri_kwargs = self.loss_kwargs.get('tri')
                loss_dict['loss_triplet'] = triplet_loss(
                    pred_features,
                    gt_labels,
                    tri_kwargs.get('margin'),
                    tri_kwargs.get('norm_feat'),
                    tri_kwargs.get('hard_mining')
                ) * tri_kwargs.get('scale')

            if 'CircleLoss' in loss_names:
                circle_kwargs = self.loss_kwargs.get('circle')
                loss_dict['loss_circle'] = pairwise_circleloss(
                    pred_features,
                    gt_labels,
                    circle_kwargs.get('margin'),
                    circle_kwargs.get('gamma')
                ) * circle_kwargs.get('scale')

            if 'Cosface' in loss_names:
                cosface_kwargs = self.loss_kwargs.get('cosface')
                loss_dict['loss_cosface'] = pairwise_cosface(
                    pred_features,
                    gt_labels,
                    cosface_kwargs.get('margin'),
                    cosface_kwargs.get('gamma'),
                ) * cosface_kwargs.get('scale')

        elif opt['type'] == 'mtrain':

            loss_dict['loss_triplet_add'] = triplet_loss_Meta(
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

            loss_dict['loss_triplet_mtrain'] = triplet_loss_Meta(
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

            loss_dict['loss_stc'] = domain_SCT_loss(
                pred_features,
                domain_labels,
                True,
                'cosine_sim',
            )

        elif opt['type'] == 'mtest':

            loss_dict['loss_triplet_mtest'] = triplet_loss_Meta(
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

        else:
            raise NotImplementedError


        return loss_dict

if __name__ == "__main__":

    # 配置
    import argparse
    parser = argparse.ArgumentParser(description="fastreid Training")
    parser.add_argument("--config-file", default="./configs/bagtricks_DR50_mix.yml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", type=bool, default=True, help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    from fastreid.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    from fastreid.engine import default_setup
    default_setup(cfg, args)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES1 = 600
    cfg.MODEL.HEADS.NUM_CLASSES2 = 600
    cfg.MODEL.HEADS.NUM_CLASSES3 = 600

    # 初始化模型
    # from pytorch_fastreid.modeling.meta_arch.baseline import Baseline as build_pytorch_model
    # from pytorch_fastreid.modeling.meta_arch.build import build_model as build_pytorch_model
    pt_model = Baseline(cfg).eval()
    print(pt_model)
