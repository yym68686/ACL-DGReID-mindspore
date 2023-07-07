import torch

# 创建两个形状不同的张量
x = torch.randn(2, 1)
y = torch.randn(2, 3)

# 使用expand_as函数将x扩展为与y具有相同形状的张量
# expanded_x = x.expand_as(torch.empty(2, 3))
expanded_x = x.expand_as(y)

print(x)  # 输出: torch.Size([2, 1])
print(y)  # 输出: torch.Size([2, 3])
print(expanded_x)  # 输出: torch.Size([2, 3])