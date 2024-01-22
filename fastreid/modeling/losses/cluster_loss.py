# import torch
# import torch.nn.functional as F
import mindspore

def intraCluster(feature, domain_ids, margin=0.1):
    # feature = F.normalize(feature, 2, 1)
    feature = mindspore.ops.L2Normalize(axis=1)(feature)
    loss = 0
    count = 0
    for i in range(3):
        if (domain_ids == i).sum().item() == 0:
            continue
        count += 1
        # QUES
        # loss += 0.1
        # print(2222222222222222)
        # print("feature", feature)
        # print("domain_ids==i", domain_ids==i)
        # print("feature[domain_ids==i]", feature[domain_ids==i])
        # print("feature[domain_ids==i].mean()", feature[domain_ids==i].mean())
        # print("feature[domain_ids==i] - feature[domain_ids==i].mean()", feature[domain_ids==i] - feature[domain_ids==i].mean())
        loss += (mindspore.ops.relu(mindspore.ops.norm(feature[domain_ids==i] - feature[domain_ids==i].mean(), 2, 1) - margin) ** 2).mean()
        # print(33333333333333333333333)
        # loss += (F.relu(torch.norm(feature[domain_ids==i] - feature[domain_ids==i].mean(), 2, 1) - margin) ** 2).mean()

    return loss / count


def interCluster(feature, domain_ids, margin=0.3):
    feature = mindspore.ops.L2Normalize(axis=1)(feature)
    # feature = F.normalize(feature, 2, 1)
    candidate_list = []
    loss = 0
    count = 0
    # print("domain_ids", type(domain_ids), domain_ids.shape, domain_ids)
    for i in range(3):
        # QUES
        # print("domain_ids", type(domain_ids), domain_ids)
        if (domain_ids == i).sum().item() == 0:
            continue
        candidate_list.append(feature[domain_ids==i].mean())
        # candidate_list.append(1)
    for i in range(len(candidate_list)):
        for j in range(i+1, len(candidate_list)):
            count += 1
            loss += mindspore.ops.relu(margin - mindspore.ops.norm(mindspore.Tensor(candidate_list[i]-candidate_list[j]), 2, 0)) ** 2
            # loss += F.relu(margin - torch.norm(candidate_list[i]-candidate_list[j], 2, 0)) ** 2
    
    return loss / count if count else domain_ids.float().mean()