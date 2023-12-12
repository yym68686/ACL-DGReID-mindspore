# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
# import torch
# import torch.nn.functional as F
import mindspore

from fastreid.utils.events import get_event_storage


def log_accuracy(pred_class_logits, gt_classes, topk=(1,)):
    """
    Log the accuracy metrics to EventStorage.
    """
    bsz = pred_class_logits.shape[0]
    # bsz = pred_class_logits.size(0)
    maxk = max(topk)
    # print("pred_class_logits", type(pred_class_logits), pred_class_logits)
    _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
    pred_class = pred_class.t()
    correct = pred_class.equal(gt_classes.view(1, -1).expand_as(pred_class))
    # correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))
    # print("gt_classes", type(gt_classes), gt_classes)
    # print("pred_class", type(pred_class), pred_class)
    # correct = pred_class.equal(gt_classes.expand_as(pred_class))
    # print("correct", type(correct), correct)
    # correct = mindspore.ops.equal(pred_class, gt_classes.view(1, -1).expand_as(pred_class))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(axis=0, keepdims=True)

        # correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        # ret.append(correct_k.mul_(1. / bsz))

        # correct = mindspore.ops.mul(correct_k, 1. / bsz)
        ret.append(correct_k.mul(1. / bsz))

    storage = get_event_storage()
    storage.put_scalar("cls_accuracy", ret[0])


def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2):
    # print("pred_class_outputs.shape", pred_class_outputs.shape)
    # pred_class_outputs = pred_class_outputs.squeeze(0)
    num_classes = pred_class_outputs.shape[1]
    # num_classes = pred_class_outputs.size(1)

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = mindspore.ops.softmax(pred_class_outputs, axis=1)
        smooth_param = alpha * soft_label[mindspore.ops.arange(soft_label.shape[0]), gt_classes].unsqueeze(1)
        # soft_label = F.softmax(pred_class_outputs, dim=1)
        # smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

    log_probs = mindspore.ops.log_softmax(pred_class_outputs, axis=1)
    # log_probs = F.log_softmax(pred_class_outputs, dim=1)
    targets = mindspore.ops.ones_like(log_probs)
    # print("targets", type(targets), targets.shape)
    # print("smooth_param", type(smooth_param), smooth_param)
    # print("smooth_param / (num_classes - 1)", smooth_param / (num_classes - 1))
    targets *= smooth_param / (num_classes - 1)
    # print("targets", type(targets), targets.shape)
    # print("gt_classes", type(gt_classes), gt_classes)
    # print("gt_classes", type(gt_classes), gt_classes, gt_classes.unsqueeze(0).shape, gt_classes.unsqueeze(1).shape)
    kkk = mindspore.Tensor(1 - smooth_param, dtype=mindspore.float32).unsqueeze(0).unsqueeze(1)
    concat = kkk.tile((gt_classes.shape[0], 1))
    concat = mindspore.Tensor(concat, dtype=mindspore.float32)
    # print("concat", type(concat), concat.shape)
    # print("1 - smooth_param", type(1 - smooth_param), 1 - smooth_param, mindspore.Tensor(1 - smooth_param, dtype=mindspore.float32).unsqueeze(0).shape)
    targets = mindspore.ops.tensor_scatter_elements(input_x=targets, axis=1, indices=gt_classes.unsqueeze(1), updates=concat)
    # targets = mindspore.ops.tensor_scatter_elements(input_x=targets, axis=1, indices=gt_classes.unsqueeze(0).unsqueeze(0), updates=mindspore.Tensor(1 - smooth_param, dtype=mindspore.float32).unsqueeze(0).unsqueeze(0))
    # with torch.no_grad():
    #     targets = torch.ones_like(log_probs)
    #     targets *= smooth_param / (num_classes - 1)
    #     targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

    targets = mindspore.Tensor(targets, dtype=mindspore.float32)
    # print("targets", type(targets), targets.shape)
    log_probs = mindspore.Tensor(log_probs, dtype=mindspore.float32)
    # print("log_probs", type(log_probs), log_probs.shape)
    loss = (-targets * log_probs).sum(axis=1)
    # loss = (-targets * log_probs).sum(dim=1)

    non_zero_cnt = max(mindspore.ops.nonzero(loss).shape[0], 1)
    # with torch.no_grad():
    #     non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

    loss = loss.sum() / non_zero_cnt

    return loss