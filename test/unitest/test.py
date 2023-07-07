import unittest
import torch
import mindspore
import mindspore.ops as ops
import os
os.system("clear")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fastreid.modeling.ops as test_ops_mindspore
import fastreid.modeling.backbones.meta_dynamic_router_resnet as test_meta_dynamic_router_resnet_mindspore
import pytorch_function as test_pytorch

class TestBackbones(unittest.TestCase):
    def setUp(self):
        print("\r=>", self._testMethodName[5:])

    def tearDown(self):
        pass

    def test_MetaConv2d(self):
        in_channels = 3

        input_tensor = ops.randn(1, in_channels, 32, 32)
        model = test_ops_mindspore.MetaConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_tensor = model(input_tensor)

        input_tensor = torch.randn(1, in_channels, 32, 32)
        model = test_pytorch.MetaConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        expected_tensor = model(input_tensor)
        # print(output_tensor, expected_tensor)

        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    def test_MetaBNNorm(self):
        # 输入张量的形状应该是(N, C, H, W)，其中 N 是批次大小，C 是通道数，H 是高度，W 是宽度。num_features 应该等于 C，否则会报错。
        num_features = 3

        input_tensor = ops.randn(1, num_features, 32, 32)
        model = test_ops_mindspore.MetaBNNorm(num_features)
        output_tensor = model(input_tensor)

        input_tensor = torch.randn(1, num_features, 32, 32)
        model = test_pytorch.MetaBNNorm(num_features)
        expected_tensor = model(input_tensor)

        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    @unittest.skip("InstanceNorm2d 只支持 GPU 上运行")
    def test_MetaINNorm(self):
        num_features = 3
        mindspore.set_context(device_target="GPU")

        input_tensor = ops.randn(1, num_features, 32, 32)
        model = test_ops_mindspore.MetaINNorm(num_features)
        output_tensor = model(input_tensor)

        input_tensor = torch.randn(1, num_features, 32, 32)
        model = test_pytorch.MetaINNorm(num_features)
        expected_tensor = model(input_tensor)

        self.assertEqual(output_tensor.shape, expected_tensor.shape)


    # def test_MetaIBNNorm(self):
    #     planes = 64
    #     num_features = 3
    #     input_tensor = ops.randn(1, num_features, 32, 32)
    #     model = test_ops_mindspore.MetaIBNNorm(planes)
    #     output_tensor = model(input_tensor)
    #     input_tensor = torch.randn(1, num_features, 32, 32)
    #     model = test_pytorch.MetaIBNNorm(planes)
    #     expected_tensor = model(input_tensor)
    #     self.assertEqual(output_tensor.shape, expected_tensor.shape)

    # def test_Bottleneck2(self):
    #     num_features = 3
    #     bn_norm, with_ibn, with_se = None, False, False
    #     input_tensor = ops.randn(1, num_features, 32, 32)
    #     model = test_meta_dynamic_router_resnet_mindspore.Bottleneck2(256, 64, bn_norm, with_ibn, with_se)
    #     output_tensor = model(input_tensor)
    #     input_tensor = torch.randn(1, num_features, 32, 32)
    #     model = test_pytorch.Bottleneck2(256, 64, bn_norm, with_ibn, with_se)
    #     expected_tensor = model(input_tensor)
    #     self.assertEqual(output_tensor.shape, expected_tensor.shape)

if __name__ == '__main__':
    unittest.main()