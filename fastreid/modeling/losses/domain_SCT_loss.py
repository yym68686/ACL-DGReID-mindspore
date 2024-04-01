# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
# import torch
# import torch.nn.functional as F
import mindspore
from .utils import concat_all_gather, euclidean_dist, normalize, cosine_dist, cosine_sim


def domain_SCT_loss(embedding, domain_labels, norm_feat, type):

    # type = 'cosine' # 'cosine', 'euclidean'
    # eps=1e-05
    if norm_feat: embedding = normalize(embedding, axis=-1)
    unique_label, idx = mindspore.ops.unique(domain_labels)
    # unique_label = torch.unique(domain_labels)
    embedding_all = list()
    for i, x in enumerate(unique_label):
        embedding_all.append(embedding[x == domain_labels])
    num_domain = len(embedding_all)
    loss_all = []
    for i in range(num_domain):
        feat = embedding_all[i]
        center_feat = mindspore.ops.mean(feat, 0)
        # center_feat = torch.mean(feat, 0)
        if type == 'euclidean':
            loss = mindspore.ops.mean(euclidean_dist(center_feat.view(1, -1), feat))
            # loss = torch.mean(euclidean_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cosine':
            loss = mindspore.ops.mean(cosine_dist(center_feat.view(1, -1), feat))
            # loss = torch.mean(cosine_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cosine_sim':
            loss = mindspore.ops.mean(cosine_sim(center_feat.view(1, -1), feat))
            # loss = torch.mean(cosine_sim(center_feat.view(1, -1), feat))
            loss_all.append(loss)

    loss_all = mindspore.ops.mean(mindspore.ops.stack(loss_all))
    # loss_all = torch.mean(torch.stack(loss_all))

    return loss_all
