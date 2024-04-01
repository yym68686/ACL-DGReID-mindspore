# In MindSpore
import mindspore

x = mindspore.Tensor([1, 3, 2, 3], mindspore.float32)
output, idx = mindspore.ops.unique(x)
print(output)
# [1. 3. 2.]
print(idx)
# [0 1 2 1]

# In PyTorch
import torch

output, inverse_indices, counts = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long), sorted=True, return_inverse=True, return_counts=True)
print(output)
# tensor([1, 2, 3])
print(inverse_indices)
# tensor([0, 2, 1, 2])
print(counts)
# tensor([1, 1, 2])

output, inverse_indices = torch.unique(torch.tensor([1, 3, 2, 3], dtype=torch.long), return_inverse=True)
print(output)
print(inverse_indices)

# # Example of using unique with dim
# output, inverse_indices = torch.unique(torch.tensor([[3, 1], [1, 2]], dtype=torch.long), sorted=True, return_inverse=True, dim=0)
# print(output)
# # tensor([[1, 2],
# #         [3, 1]])
# print(inverse_indices)
# # tensor([1, 0])
