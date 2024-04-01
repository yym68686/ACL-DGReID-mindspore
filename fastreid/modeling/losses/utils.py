# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# import torch
import mindspore
# import torch.nn.functional as F


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (mindspore.ops.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    # x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    # m, n = x.shape[0], y.shape[0]
    # # m, n = x.size(0), y.size(0)
    # xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    # dist = xx + yy - 2 * torch.matmul(x, y.t())
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # return dist

    m, n = x.shape[0], y.shape[0]
    # m, n = x.size(0), y.size(0)
    xx = mindspore.ops.sum(mindspore.ops.pow(x, 2), 1, keepdim=True).broadcast_to((m, n))
    yy = mindspore.ops.sum(mindspore.ops.pow(y, 2), 1, keepdim=True).broadcast_to((n, m)).t()
    dist = xx + yy - 2 * mindspore.ops.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2 - 2 * torch.mm(x, y.t())
    return dist


def cosine_sim(x, y):
    bs1, bs2 = x.shape[0], y.shape[0]
    # bs1, bs2 = x.size(0), y.size(0)
    frac_up = mindspore.ops.matmul(x, y.transpose(0, 1))
    # frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (mindspore.ops.sqrt(mindspore.ops.sum(mindspore.ops.pow(x, 2), 1))).view(bs1, 1).tile((1, bs2)) * \
                (mindspore.ops.sqrt(mindspore.ops.sum(mindspore.ops.pow(y, 2), 1))).view(1, bs2).tile((bs1, 1))
    # frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
    #             (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return cosine