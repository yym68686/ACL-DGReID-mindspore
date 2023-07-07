import unittest
import torch
import mindspore
import mindspore.ops as ops
import os
os.system("clear")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fastreid.modeling.ops as test_mindspore
import pytorch_function as test_pytorch
# import pytorch_fastreid.modeling.ops as test_pytorch

class TestBackbones(unittest.TestCase):
    def setUp(self):
        print("\r=>", self._testMethodName[5:])

    def tearDown(self):
        pass

    def test_MetaConv2d(self):
        in_channels = 3
        input_tensor = ops.randn(1, in_channels, 32, 32)
        model = test_mindspore.MetaConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_tensor = model(input_tensor)
        input_tensor = torch.randn(1, in_channels, 32, 32)
        model = test_pytorch.MetaConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        expected_tensor = model(input_tensor)
        # print(output_tensor, expected_tensor)
        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    def test_MetaBNNorm(self):
        # 输入张量的形状应该是(N, C, H, W)，其中 N 是批次大小，C 是通道数，H 是高度，W 是宽度。num_features 应该等于 C，否则会报错。
        num_features = 3
        input_tensor = torch.randn(1, num_features, 32, 32)
        model = test_pytorch.MetaBNNorm(num_features)
        expected_tensor = model(input_tensor)
        input_tensor = ops.randn(1, num_features, 32, 32)
        model = test_mindspore.MetaBNNorm(num_features)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, expected_tensor.shape)

if __name__ == '__main__':
    unittest.main()