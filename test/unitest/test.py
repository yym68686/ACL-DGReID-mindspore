import unittest
import torch
import mindspore
import mindspore.ops as ops
import os
os.system("clear")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fastreid.modeling.ops as test_ops_mindspore
import fastreid.layers.batch_norm as test_batch_norm_mindspore
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

    def test_MetaIBNNorm(self):
        # num_features 必须是偶数
        num_features = planes = 4
        input_tensor = ops.randn(1, num_features, 32, 32)
        model = test_ops_mindspore.MetaIBNNorm(planes)
        output_tensor = model(input_tensor)
        input_tensor = torch.randn(1, num_features, 32, 32)
        model = test_pytorch.MetaIBNNorm(planes)
        expected_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    def test_Bottleneck2(self):
        # num_features 必须是 out_channel 四倍
        num_features = 512
        # out_channel 必须大于 4
        out_channel = 128
        K = 4
        bn_norm, with_ibn, with_se = None, False, False
        input_tensor = ops.randn(1, num_features, 32, 32).tile((1, K, 1, 1))
        model = test_meta_dynamic_router_resnet_mindspore.Bottleneck2(num_features, out_channel, bn_norm, with_ibn, with_se)
        output_tensor = model(input_tensor)
        input_tensor = torch.randn(1, num_features, 32, 32).repeat(1, K, 1, 1)
        model = test_pytorch.Bottleneck2(num_features, out_channel, bn_norm, with_ibn, with_se)
        expected_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    @unittest.skip("从未使用过，不做测试")
    def test_BasicBlock(self):
        pass

    def test_Bottleneck(self):
        bn_norm, with_ibn, with_se = 'IN', False, False
        inplanes = num_features = 256
        planes = 64
        input_tensor = ops.randn(1, num_features, 32, 32)
        model = test_meta_dynamic_router_resnet_mindspore.Bottleneck(inplanes, planes, bn_norm, with_ibn, with_se)
        output_tensor = model(input_tensor)
        input_tensor = torch.randn(1, num_features, 32, 32)
        model = test_pytorch.Bottleneck(256, 64, 'IN', False, with_se)
        expected_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    def test_MetaLinear(self):
        in_channels = 3
        out_channels = 4

        input_tensor = ops.randn(1, in_channels)
        model = test_ops_mindspore.MetaLinear(in_channels, out_channels, has_bias=True)
        output_tensor = model(input_tensor)

        input_tensor = torch.randn(1, in_channels)
        model = test_pytorch.MetaLinear(in_channels, out_channels, bias=True)
        expected_tensor = model(input_tensor)

        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    def test_MetaSELayer(self):
        in_channels = 16

        input_tensor = ops.randn(1, in_channels, 32, 32)
        model = test_meta_dynamic_router_resnet_mindspore.MetaSELayer(in_channels)
        output_tensor = model(input_tensor)

        input_tensor = torch.randn(1, in_channels, 32, 32)
        model = test_pytorch.MetaSELayer(in_channels)
        expected_tensor = model(input_tensor)

        self.assertEqual(output_tensor.shape, expected_tensor.size())

    @unittest.skip("InstanceNorm2d 只支持 GPU 上运行")
    def test_HyperRouter(self):
        planes = in_channels = 256

        input_tensor = ops.randn(1, planes, 32, 32)
        model = test_meta_dynamic_router_resnet_mindspore.HyperRouter(planes)
        output_tensor = model(input_tensor)

        input_tensor = torch.randn(1, planes, 32, 32)
        model = test_pytorch.HyperRouter(planes)
        expected_tensor = model(input_tensor)

        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    def test_MetaGate(self):
        feat_dim = in_channels = 256

        input_tensor = ops.randn(1, feat_dim, 32, 32)
        model = test_ops_mindspore.MetaGate(feat_dim)
        output_tensor = model(input_tensor, input_tensor)

        input_tensor = torch.randn(1, feat_dim, 32, 32)
        model = test_pytorch.MetaGate(feat_dim)
        expected_tensor = model(input_tensor, input_tensor)

        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    def test_Identity(self):
        in_channels = 256
        input_tensor = ops.randn(1, in_channels, 32, 32)
        model = test_meta_dynamic_router_resnet_mindspore.Identity()
        output_tensor = model(input_tensor)

        input_tensor = torch.randn(1, in_channels, 32, 32)
        model = test_pytorch.Identity()
        expected_tensor = model(input_tensor)

        self.assertEqual(output_tensor[0].shape, expected_tensor[0].shape)

    @unittest.skip("wait")
    def test_ResNet(self):
        # fmt: off
        pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
        pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
        last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
        bn_norm       = cfg.MODEL.BACKBONE.NORM
        with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
        with_se       = cfg.MODEL.BACKBONE.WITH_SE
        with_nl       = cfg.MODEL.BACKBONE.WITH_NL
        depth         = cfg.MODEL.BACKBONE.DEPTH
        # fmt: on

        num_blocks_per_stage = {
            '18x': [2, 2, 2, 2],
            '34x': [3, 4, 6, 3],
            '50x': [3, 4, 6, 3],
            '101x': [3, 4, 23, 3],
        }[depth]

        nl_layers_per_stage = {
            '18x': [0, 0, 0, 0],
            '34x': [0, 0, 0, 0],
            '50x': [0, 2, 3, 0],
            '101x': [0, 2, 9, 0]
        }[depth]

        block = {
            '18x': BasicBlock,
            '34x': BasicBlock,
            '50x': Bottleneck,
            '101x': Bottleneck
        }[depth]
        input_tensor = ops.randn(1, in_channels, 32, 32)
        model = test_meta_dynamic_router_resnet_mindspore.ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, block, num_blocks_per_stage, nl_layers_per_stage)
        output_tensor = model(input_tensor)
        input_tensor = torch.randn(1, in_channels, 32, 32)
        model = test_pytorch.ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, block, num_blocks_per_stage, nl_layers_per_stage)
        expected_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    @unittest.skip("从未使用过，不做测试")
    def test_IBN(self):
        planes = in_channels = 256
        bn_norm = None
        input_tensor = ops.randn(1, in_channels, 32, 32)
        model = test_batch_norm_mindspore.IBN(planes, bn_norm)
        output_tensor = model(input_tensor)
        input_tensor = torch.randn(1, in_channels, 32, 32)
        model = test_pytorch.IBN(planes, bn_norm)
        expected_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, expected_tensor.shape)

    @unittest.skip("InstanceNorm2d 只支持 GPU 上运行")
    def test_MetaParam(self):
        pass

if __name__ == '__main__':
    unittest.main()