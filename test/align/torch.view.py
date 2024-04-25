import torch

# 创建一个形状为(2, 3)的张量
x = torch.arange(6).view(1, 6)
print("Original tensor:")
print(x)
# tensor([[0, 1, 2],
#         [3, 4, 5]])

# 使用view改变张量形状为(3, 2)
y = x.view(1, 1)
print("Reshaped tensor:")
print(y)
# tensor([[0, 1],
#         [2, 3],
#         [4, 5]])

# 使用view将张量展平为一维
z = x.view(-1)
print("Flattened tensor:")
print(z)
# tensor([0, 1, 2, 3, 4, 5])