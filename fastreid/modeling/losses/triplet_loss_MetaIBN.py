# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# import torch
# import torch.nn.functional as F
import copy
import mindspore

from fastreid.utils import comm
from .utils import concat_all_gather, euclidean_dist, normalize, cosine_dist


def softmax_weights(dist, mask):
    max_v = mindspore.ops.max(dist * mask, axis=1, keepdims=True)[0]
    # max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = mindspore.ops.sum(mindspore.ops.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    # Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = mindspore.ops.exp(diff) * mask / Z
    # W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.shape) == 2
    # assert len(dist_mat.size()) == 2
    N = dist_mat.shape[0]
    # N = dist_mat.size(0)

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]

    # try:

    # dist_ap, relative_p_inds = torch.max(
    #     dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # dist_ap = dist_ap.squeeze(1)

    #
    dist_ap = list()
    dist_an = list()
    for i in range(dist_mat.shape[0]):
        for item in mindspore.ops.max(dist_mat[i][is_pos[i]]):
            dist_ap.append(mindspore.Tensor(item, mindspore.float32))
        for item in mindspore.ops.min(dist_mat[i][is_neg[i]]):
            dist_an.append(mindspore.Tensor(item, mindspore.float32))
    # print("dist_mat", dist_mat)
    # print("is_pos", is_pos)
    # print("is_neg", is_neg)
    # print("dist_ap 1", dist_ap)
    dist_ap = mindspore.ops.stack(dist_ap)
    # print("dist_ap 2", dist_ap)
    # print("dist_an 1", dist_an)
    dist_an = mindspore.ops.stack(dist_an)
    # print("dist_an 2", dist_an)


    # dist_ap = list()
    # for i in range(dist_mat.shape[0]):
    #     # print("is_neg[i]", is_neg[i])
    #     # print("dist_mat[i]", dist_mat[i])
    #     # print("dist_mat[i][is_neg[i]]", dist_mat[i][is_neg[i]])
    #     dist_ap.append(mindspore.ops.max(dist_mat[i][is_pos[i]]))
    #     # dist_ap.append(torch.max(dist_mat[i][is_pos[i]]))
    # dist_ap = mindspore.ops.stack(dist_ap)
    # # is_neg = mindspore.Tensor(copy.deepcopy(is_neg))
    # # is_neg = mindspore.Tensor(is_neg.asnumpy())
    # # print("is_neg", type(is_neg), is_neg)
    # # dist_ap = torch.stack(dist_ap)

    # dist_an = list()
    # for i in range(dist_mat.shape[0]):
    #     # print("is_neg", is_neg)
    #     # print("is_neg[i]", is_neg[i])
    #     # print("dist_mat[i]", dist_mat[i])
    #     # print("dist_mat[i][is_neg[i]]", dist_mat[i][is_neg[i]])
    #     dist_an.append(mindspore.ops.min(dist_mat[i][is_neg[i]]))
    #     # dist_an.append(torch.min(dist_mat[i][is_neg[i]]))
    # dist_an = mindspore.ops.stack(dist_an)


    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]

    # dist_an = torch.stack(dist_an)
    # dist_an, relative_n_inds = torch.min(
    #     dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # dist_an = dist_an.squeeze(1)
    # except:
    #     print('.')



    # shape [N]

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.shape) == 2
    # assert len(dist_mat.size()) == 2

    is_pos = is_pos.float()
    is_neg = is_neg.float()
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = mindspore.ops.sum(dist_ap * weights_ap, dim=1)
    dist_an = mindspore.ops.sum(dist_an * weights_an, dim=1)
    # dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    # dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def triplet_loss_Meta(embedding, targets, margin, norm_feat, hard_mining, dist_type, loss_type, domain_labels = None, pos_flag = [1,0,0], neg_flag = [0,0,1]):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    if norm_feat: embedding = normalize(embedding, axis=-1)

    # For distributed training, gather all features from different process.
    if comm.get_world_size() > 1:
        all_embedding = concat_all_gather(embedding)
        all_targets = concat_all_gather(targets)
        domain_labels = concat_all_gather(domain_labels)
    else:
        all_embedding = embedding
        all_targets = targets

    if dist_type == 'euclidean':
        dist_mat = euclidean_dist(all_embedding, all_embedding)
    elif dist_type == 'cosine':
        dist_mat = cosine_dist(all_embedding, all_embedding)

    N = dist_mat.shape[0]
    # N = dist_mat.size(0)
    if (pos_flag == [1,0,0] and neg_flag == [0,1,1]) or domain_labels == None:
        is_pos = all_targets.view(N, 1).broadcast_to((N, N)).eq(all_targets.view(N, 1).broadcast_to((N, N)).t())
        is_neg = all_targets.view(N, 1).broadcast_to((N, N)).ne(all_targets.view(N, 1).broadcast_to((N, N)).t())
        # is_pos = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t())
        # is_neg = all_targets.view(N, 1).expand(N, N).ne(all_targets.view(N, 1).expand(N, N).t())
    else:
        vec1 = copy.deepcopy(all_targets)
        for i in range(N):
            vec1[i] = i # [0,1,2,3,4,~~]
        is_same_img = vec1.broadcast_to((N, N)).eq(vec1.broadcast_to((N, N)).t())
        is_same_instance = all_targets.view(N, 1).broadcast_to((N, N)).eq(all_targets.view(N, 1).broadcast_to((N, N)).t())
        # print("is_same_instance", is_same_instance)
        is_same_domain = domain_labels.view(N, 1).broadcast_to((N, N)).eq(domain_labels.view(N, 1).broadcast_to((N, N)).t())
        # is_same_img = vec1.expand(N, N).eq(vec1.expand(N, N).t())
        # is_same_instance = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t())
        # is_same_domain = domain_labels.view(N, 1).expand(N, N).eq(domain_labels.view(N, 1).expand(N, N).t())

        set0 = is_same_img

        set_all = []
        # a = copy.deepcopy(is_same_instance)
        # print("is_same_instance 2", is_same_instance)
        # print("is_same_instance 2", is_same_instance)
        set_all.extend([is_same_instance.int() * (is_same_img == False).int()])
        # set_all.extend([is_same_instance * (is_same_img == False)])
        # print("is_same_instance 3", is_same_instance)
        # print("(is_same_instance == False).int()", mindspore.Tensor(is_same_instance == False).int())
        set_all.extend([(is_same_instance == False).int() * (is_same_domain == True).int()])
        # set_all.extend([(is_same_instance == False) * (is_same_domain == True)])
        set_all.extend([(is_same_domain == False).int()])
        # set_all.extend([is_same_domain == False])

        is_pos = copy.deepcopy(set0)
        is_neg = copy.deepcopy(set0==False)
        is_neg[:] = False
        for i, bool_flag in enumerate(pos_flag):
            if bool_flag == 1:
                is_pos += set_all[i]

        for i, bool_flag in enumerate(neg_flag):
            if bool_flag == 1:
                is_neg += set_all[i]

        # print(pos_flag)
        # print(is_pos.type(torch.IntTensor))
        # print(neg_flag)
        # print(is_neg.type(torch.IntTensor))


    if hard_mining:
        # print("dist_mat", dist_mat)
        # print("is_neg", is_neg)
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    # print(dist_an)
    y = dist_an.new_ones(dist_an.shape)
    # y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        # all(sum(is_pos) == 1)
        loss = mindspore.ops.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
        # loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        if loss_type == 'logistic':
            loss = mindspore.ops.soft_margin_loss(dist_an - dist_ap, y)
            # loss = F.soft_margin_loss(dist_an - dist_ap, y)
            # fmt: off
            if loss == float('Inf'): loss = mindspore.ops.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
            # fmt: on
        elif loss_type == 'hinge':
            loss = mindspore.ops.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
            # loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)

    return loss
