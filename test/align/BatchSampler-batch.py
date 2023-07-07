import torch
import numpy as np
import mindspore as ms
import mindspore.dataset as ds

data = [1, 2, 3, 4, 5]
batch_size = 2
drop_last = False

batch_sampler = torch.utils.data.sampler.BatchSampler(data, batch_size, drop_last)
print(type(batch_sampler))
for batch in batch_sampler:
    print(batch)

dataset = ds.NumpySlicesDataset(data, shuffle=False)
dataset = dataset.batch(batch_size, drop_last)
for data in dataset:
    print(data[0].value().asnumpy().tolist())