import mindspore as ms
import torch

sampler = [1, 2, 3, 4, 5]
dataset = ms.Tensor([1, 2, 3, 4, 5])
mini_batch_size = 2
batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, mini_batch_size, True)
print(batch_sampler)

dataset = dataset.batch(100, True)
print(dataset)