from __future__ import absolute_import

# import torch
# from torch import nn
import mindspore

# class CenterLoss(nn.Module):
#     """Center loss.

#     Reference:
#     Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

#     Args:
#         num_classes (int): number of classes.
#         feat_dim (int): feature dimension.
#     """

#     def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
#         super(CenterLoss, self).__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.use_gpu = use_gpu

#         if self.use_gpu:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#         else:
#             self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

#     def forward(self, x, labels):
#         """
#         Args:
#             x: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (num_classes).
#         """
#         assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

#         batch_size = x.size(0)
#         distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#         distmat.addmm_(1, -2, x, self.centers.t())

#         classes = torch.arange(self.num_classes).long()
#         if self.use_gpu: classes = classes.cuda()
#         labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#         mask = labels.eq(classes.expand(batch_size, self.num_classes))

#         dist = distmat * mask.float()
#         loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
#         #dist = []
#         #for i in range(batch_size):
#         #    value = distmat[i][mask[i]]
#         #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
#         #    dist.append(value)
#         #dist = torch.cat(dist)
#         #loss = dist.mean()
#         return loss

def centerLoss(distmat, labels):
    """
    Args:
        x: feature matrix with shape (batch_size, feat_dim).
        labels: ground truth labels with shape (num_classes).
    """

    batch_size = distmat.shape[0]
    num_classes = distmat.shape[1]
    # batch_size = distmat.size(0)
    # num_classes = distmat.size(1)

    # num_classes = mindspore.Tensor(num_classes, dtype=mindspore.float32)
    # print("type num_classes", type(num_classes), num_classes)
    # print("type num_classes", type(num_classes), num_classes.dtype, num_classes.shape, num_classes)
    classes = mindspore.ops.arange(0, num_classes)
    # classes = mindspore.Tensor(classes, dtype=mindspore.int64)
    # QUES
    # print("classes", type(classes), classes.shape)
    # print("labels.shape", type(labels), labels.dtype, labels)
    # labels = mindspore.Tensor(labels, mindspore.float32)
    # print("labels", type(labels), labels)
    labels = labels.unsqueeze(1).broadcast_to((batch_size, num_classes))
    # print("labels", type(labels), labels)
    # labels = labels.unsqueeze(1).broadcast_to((batch_size, num_classes))
    mask = mindspore.ops.equal(labels, classes.broadcast_to((batch_size, num_classes)))
    # print("mask", type(mask), mask.shape)
    # mask = labels.eq(classes.broadcast_to((batch_size, num_classes)))

    # classes = torch.arange(num_classes).long().to(distmat.device)
    # labels = labels.unsqueeze(1).expand(batch_size, num_classes)
    # mask = labels.eq(classes.expand(batch_size, num_classes))

    # import pdb; pdb.set_trace()

    # QUES
    # loss = 0.1
    # dist = distmat * mask
    # distmat = mindspore.Tensor(distmat, mindspore.float32)
    # print("distmat", type(distmat), distmat.shape)
    # mask = mindspore.Tensor(mask, mindspore.float32)
    # print("mask", type(mask), mask.shape)
    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
    
    return loss


if __name__ == '__main__':
    use_gpu = False
    # center_loss = CenterLoss(use_gpu=use_gpu)
    center_loss = centerLoss
    features = mindspore.ops.rand(16, 2048)
    targets = mindspore.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    # if use_gpu:
    #     features = mindspore.ops.rand(16, 2048).cuda()
    #     targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()

    loss = center_loss(features, targets)
    print(loss)