import unittest
import numpy as np
import re
import torch
from collections import OrderedDict
import mindspore
import mindspore.ops as ops
mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU", device_id=0)
# mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")
import os
os.system("clear")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fastreid.modeling.ops as test_ops_mindspore
import fastreid.layers.batch_norm as test_batch_norm_mindspore
import fastreid.modeling.backbones.meta_dynamic_router_resnet as test_meta_dynamic_router_resnet_mindspore
import pytorch_fastreid.modeling.backbones.meta_dynamic_router_resnet as test_meta_dynamic_router_resnet_pytorch
import pytorch_function as test_pytorch

class TestBackbones(unittest.TestCase):
    def setUp(self):
        print("\r=>", self._testMethodName[5:])

    def tearDown(self):
        pass

    def test_ResNet(self):

        # 初始化模型
        ms_model = test_meta_dynamic_router_resnet_mindspore.build_meta_dynamic_router_resnet_backbone().set_train(False)
        pt_model = test_meta_dynamic_router_resnet_pytorch.build_meta_dynamic_router_resnet_backbone(1).eval()

        # 得到所有网络层和值的有序字典
        mindspore_model_dict = OrderedDict()
        for item in ms_model.get_parameters():
            mindspore_model_dict[item.name] = item.value()
        pytorch_model_dict = pt_model.state_dict()

        # 参数映射
        from pth2ckpt import param_convert
        Parameter_map = param_convert(mindspore_model_dict, pytorch_model_dict)

        # 权重迁移
        for item in Parameter_map.keys():
            pt_parameter = eval("pt_model.{}.detach().numpy().reshape((-1,))[:10]".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', item))))
            ms_parameter = eval("ms_model.{}.data.asnumpy().reshape((-1,))[:10]".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', Parameter_map[item]))))
            if np.allclose(pt_parameter, ms_parameter, atol=1e-5) != True:
                pt_parameter_all = eval("pt_model.{}.detach().numpy()".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', item))))
                if mindspore_model_dict[Parameter_map[item]].shape == pt_parameter_all.shape:
                    mindspore_model_dict[Parameter_map[item]] = mindspore.Tensor(pt_parameter_all)
                else:
                    print("error: The parameter shapes are different!")

        mindspore.save_checkpoint([{"name": key, "data": mindspore.Tensor(value.numpy())} for key, value in mindspore_model_dict.items()], "/home/yuming/.cache/torch/checkpoints/ACL-DGReID.ckpt")
        print(3)
        incompatible = mindspore.load_checkpoint("/home/yuming/.cache/torch/checkpoints/ACL-DGReID.ckpt", ms_model)

        # # 权重验证
        # for item in Parameter_map.keys():
        #     pt_parameter = eval("pt_model.{}.detach().numpy().reshape((-1,))[:10]".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', item))))
        #     ms_parameter = eval("ms_model.{}.data.asnumpy().reshape((-1,))[:10]".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', Parameter_map[item]))))
        #     print(f"========= pt_model {item} ==========")
        #     print(pt_parameter)
        #     print(f"========= ms_model {item} ==========")
        #     print(ms_parameter)


        # 网络对齐

        # pt_Parameter_list_for_each_layer = []
        # ms_Parameter_list_for_each_layer = []
        # # 定义回调函数
        # def pt_hook_fn(module, input, output):
        #     # print("Layer name:", module)
        #     # print("Input shape:", input[0].shape)
        #     if "HyperRouter" in str(module) or "ResNet" in str(module):
        #         pt_Parameter_list_for_each_layer.append({"name": str(module), "output": output[0].detach().numpy().reshape((-1,))[:10]})
        #         # print("Output shape:", output[0].detach().numpy().reshape((-1,))[:10])
        #     else:
        #         pt_Parameter_list_for_each_layer.append({"name": str(module), "output": output.detach().numpy().reshape((-1,))[:10]})
        #         # print("Output shape:", output.shape)

        # def ms_hook_fn(module, input, output):
        #     if "HyperRouter" in str(module) or "ResNet" in str(module):
        #         ms_Parameter_list_for_each_layer.append({"name": str(module), "output": output[0].asnumpy().reshape((-1,))[:10]})
        #     else:
        #         ms_Parameter_list_for_each_layer.append({"name": str(module), "output": output.asnumpy().reshape((-1,))[:10]})

        # # 注册回调函数到每一层
        # for module in pt_model.modules():
        #     module.register_forward_hook(pt_hook_fn)
        # for _, cell in ms_model.cells_and_names():
        #     cell.register_forward_hook(ms_hook_fn)

        in_channels = 3
        epoch = 5
        batch_size = 8
        length = width = height = 16

        input_tensor = ops.randn(batch_size, in_channels, length, length)
        output_tensor = ms_model(input_tensor, epoch)
        input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
        expected_tensor = pt_model(input_tensor, epoch)
        # print(output_tensor[0].numpy().astype(np.float32).reshape((-1,))[:10])
        # print(expected_tensor[0].detach().numpy().astype(np.float32).reshape((-1,))[:10])
        # print(len(pt_Parameter_list_for_each_layer), len(ms_Parameter_list_for_each_layer))
        # maxnum = len(pt_Parameter_list_for_each_layer) if len(pt_Parameter_list_for_each_layer) < len(ms_Parameter_list_for_each_layer) else len(ms_Parameter_list_for_each_layer)
        # for index in range(maxnum):
        #     print(pt_Parameter_list_for_each_layer[index]["name"])
        #     print(pt_Parameter_list_for_each_layer[index]["output"])
        #     print(ms_Parameter_list_for_each_layer[index]["name"])
        #     print(ms_Parameter_list_for_each_layer[index]["output"])
        #     print()
        #     if np.allclose(pt_Parameter_list_for_each_layer[index]["output"], ms_Parameter_list_for_each_layer[index]["output"], atol=1e-5) == False:
        #         break

        self.assertEqual(np.allclose(output_tensor[0].numpy().astype(np.float32), expected_tensor[0].detach().numpy().astype(np.float32), atol=1e-5), True)

    # def test_MetaConv2d(self):
    #     in_channels = 3
    #     length = width = height = 16
    #     input_tensor = ops.randn(1, in_channels, length, length)
    #     model = test_ops_mindspore.MetaConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.MetaConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(output_tensor.shape, expected_tensor.shape)

    # def test_MetaBNNorm(self):
    #     # 输入张量的形状应该是(N, C, H, W)，其中 N 是批次大小，C 是通道数，H 是高度，W 是宽度。num_features 应该等于 C，否则会报错。
    #     num_features = 4
    #     length = width = height = 2

    #     input_tensor = ops.randn(1, num_features, length, length)
    #     model = test_ops_mindspore.MetaBNNorm(num_features)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.MetaBNNorm(num_features)
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # def test_MetaINNorm(self):
    #     num_features = 3
    #     length = width = height = 2

    #     input_tensor = ops.randn(1, num_features, length, length)
    #     model = test_ops_mindspore.MetaINNorm(num_features)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.MetaINNorm(num_features)
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # def test_MetaIBNNorm(self):
    #     # num_features 必须是偶数
    #     num_features = planes = 4
    #     length = width = height = 2

    #     input_tensor = ops.randn(1, num_features, length, length)
    #     model = test_ops_mindspore.MetaIBNNorm(planes)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.MetaIBNNorm(planes)
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # def test_Bottleneck2(self):
    #     # num_features 必须是 out_channel 四倍
    #     num_features = 512
    #     # out_channel 必须大于 4
    #     out_channel = 128
    #     length = width = height = 2
    #     K = 4
    #     bn_norm, with_ibn, with_se = None, False, False
    #     input_tensor = ops.randn(1, num_features, length, length).tile((1, K, 1, 1))
    #     model = test_meta_dynamic_router_resnet_mindspore.Bottleneck2(num_features, out_channel, bn_norm, with_ibn, with_se)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.Bottleneck2(num_features, out_channel, bn_norm, with_ibn, with_se)
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # @unittest.skip("从未使用过，不做测试")
    # def test_BasicBlock(self):
    #     pass

    # def test_Bottleneck(self):
    #     bn_norm, with_ibn, with_se = 'IN', False, False
    #     inplanes = num_features = 256
    #     planes = 64
    #     length = width = height = 2

    #     input_tensor = ops.randn(1, num_features, length, length)
    #     model = test_meta_dynamic_router_resnet_mindspore.Bottleneck(inplanes, planes, bn_norm, with_ibn, with_se)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.Bottleneck(256, 64, 'IN', False, with_se)
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # def test_MetaLinear(self):
    #     in_channels = 3
    #     out_channels = 4

    #     input_tensor = ops.randn(1, in_channels)
    #     model = test_ops_mindspore.MetaLinear(in_channels, out_channels, has_bias=False)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.MetaLinear(in_channels, out_channels, bias=False)
    #     expected_tensor = model(input_tensor)
    #     print(output_tensor.numpy().astype(np.float32), "\n\n\n", expected_tensor.detach().numpy().astype(np.float32))
    #     # result = output_tensor[0].numpy().astype(np.float32) - expected_tensor[0].detach().numpy().astype(np.float32)
    #     # print(result)

    #     # self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # def test_MetaSELayer(self):
    #     in_channels = 16
    #     length = width = height = 2

    #     input_tensor = ops.randn(1, in_channels, length, length)
    #     model = test_meta_dynamic_router_resnet_mindspore.MetaSELayer(in_channels)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.MetaSELayer(in_channels)
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # @unittest.skip("InstanceNorm2d 只支持 GPU 上运行")
    # def test_HyperRouter(self):
    #     planes = in_channels = 256
    #     length = width = height = 2

    #     input_tensor = ops.randn(1, planes, length, length)
    #     model = test_meta_dynamic_router_resnet_mindspore.HyperRouter(planes)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.HyperRouter(planes)
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # def test_MetaGate(self):
    #     feat_dim = in_channels = 256
    #     length = width = height = 2

    #     input_tensor = ops.randn(1, feat_dim, length, length)
    #     model = test_ops_mindspore.MetaGate(feat_dim)
    #     output_tensor = model(input_tensor, input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.MetaGate(feat_dim)
    #     expected_tensor = model(input_tensor, input_tensor)

    #     self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # def test_Identity(self):
    #     in_channels = 256
    #     length = width = height = 2

    #     input_tensor = ops.randn(1, in_channels, length, length)
    #     model = test_meta_dynamic_router_resnet_mindspore.Identity()
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
    #     model = test_pytorch.Identity()
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(np.allclose(output_tensor[0].numpy().astype(np.float32), expected_tensor[0].detach().numpy().astype(np.float32), atol=1e-5), True)

    # @unittest.skip("从未使用过，不做测试")
    # def test_IBN(self):
    #     planes = in_channels = 256
    #     bn_norm = None
    #     length = width = height = 8

    #     input_tensor = ops.randn(1, in_channels, length, length)
    #     model = test_batch_norm_mindspore.IBN(planes, bn_norm)
    #     output_tensor = model(input_tensor)

    #     input_tensor = torch.randn(1, in_channels, length, length)
    #     model = test_pytorch.IBN(planes, bn_norm)
    #     expected_tensor = model(input_tensor)

    #     self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    # @unittest.skip("InstanceNorm2d 只支持 GPU 上运行")
    # def test_MetaParam(self):
    #     pass

if __name__ == '__main__':
    unittest.main()