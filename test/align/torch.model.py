from torch import nn
import torch.nn.functional as F

class Convv(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义模型的层
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)

class MyModel(nn.Module):
    def __init__(self, b):
        super().__init__()
        # 定义模型的层
        self.conv = b
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

b = Convv()
a = MyModel(b)
print(a)