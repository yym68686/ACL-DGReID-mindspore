import unittest
import numpy as np
import re
import torch
from collections import OrderedDict
import mindspore
import mindspore.ops as ops
# mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU", device_id=0)
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")
import os
os.system("clear")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fastreid.modeling.ops as test_ops_mindspore
import fastreid.layers.batch_norm as test_batch_norm_mindspore
import fastreid.modeling.backbones.meta_dynamic_router_resnet as test_meta_dynamic_router_resnet_mindspore
import pytorch_fastreid.modeling.backbones.meta_dynamic_router_resnet as test_meta_dynamic_router_resnet_pytorch
import pytorch_function as test_pytorch

def get_cfg():
    # 配置
    import argparse
    parser = argparse.ArgumentParser(description="fastreid Training")
    parser.add_argument("--config-file", default="./configs/bagtricks_DR50_mix.yml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", type=bool, default=True, help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    from fastreid.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    from fastreid.engine import default_setup
    default_setup(cfg, args)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES1 = 11934
    cfg.MODEL.HEADS.NUM_CLASSES2 = 767
    cfg.MODEL.HEADS.NUM_CLASSES3 = 1041
    return cfg

cfg = get_cfg()

class TestBackbones(unittest.TestCase):
    def setUp(self):
        print("\r=>", self._testMethodName[5:])

    def tearDown(self):
        pass

    # @unittest.skip("全部网络，不做测试")
    def test_baseline(self):
        # 初始化模型
        # from pytorch_fastreid.modeling.meta_arch.baseline import Baseline as build_pytorch_model
        from pytorch_fastreid.modeling.meta_arch.build import build_model as build_pytorch_model
        pt_model = build_pytorch_model(cfg).eval()
        pt_model.to('cpu')
        # print(pt_model)

        # from fastreid.modeling.meta_arch.baseline import Baseline as build_mindspore_model
        from fastreid.modeling.meta_arch.build import build_model as build_mindspore_model
        ms_model = build_mindspore_model(cfg).set_train(False)
        # print(ms_model)
        # exit(0)

        from fastreid.utils.checkpoint import Checkpointer
        # print("pt_model baseline.weight", pt_model.heads.bottleneck[0].weight)
        Checkpointer(pt_model).load("/mnt/ssd3/yuming/model_final.pth")
        # print("pt_model baseline.weight before", pt_model.heads.bottleneck[0].weight)
        # pt_model.load_state_dict(torch.load('/mnt/ssd3/yuming/model_best.pth'))

        # 得到所有网络层和值的有序字典
        mindspore_model_dict = OrderedDict()
        for item in ms_model.get_parameters():
            mindspore_model_dict[item.name] = item.value()
        pytorch_model_dict = pt_model.state_dict()

        # 参数映射
        from pth2ckpt import param_convert
        Parameter_map = param_convert(mindspore_model_dict, pytorch_model_dict)

        # 权重迁移
        # end_num_index = 20
        for item in Parameter_map.keys():
            pt_parameter = eval("pt_model.{}.detach().cpu().numpy().reshape((-1,))[:10]".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', item))))
            ms_parameter = eval("ms_model.{}.data.asnumpy().reshape((-1,))[:10]".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', Parameter_map[item]))))
            if np.allclose(pt_parameter, ms_parameter, atol=1e-5) != True:
                pt_parameter_all = eval("pt_model.{}.detach().cpu().numpy()".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', item))))
                if mindspore_model_dict[Parameter_map[item]].shape == pt_parameter_all.shape:
                    mindspore_model_dict[Parameter_map[item]] = mindspore.Tensor(pt_parameter_all)
                else:
                    print("error: The parameter shapes are different!")

        # print("pt_model baseline.weight after", pt_model.heads.bottleneck[0].weight)

        mindspore.save_checkpoint([{"name": key, "data": mindspore.Tensor(value.numpy())} for key, value in mindspore_model_dict.items()], "./ACL-DGReID.ckpt")
        incompatible = mindspore.load_checkpoint("./ACL-DGReID.ckpt", ms_model)

        # # 权重验证
        # for item in Parameter_map.keys():
        #     pt_parameter = eval("pt_model.{}.detach().cpu().numpy().reshape((-1,))[:10]".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', item))))
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
        # maxnum = len(ms_Parameter_list_for_each_layer) if len(pt_Parameter_list_for_each_layer) < len(ms_Parameter_list_for_each_layer) else len(pt_Parameter_list_for_each_layer)
        # index = 0
        # while index < maxnum:
        #     # print(pt_Parameter_list_for_each_layer[index]["name"][:5] != ms_Parameter_list_for_each_layer[index]["name"][:5])
        #     if pt_Parameter_list_for_each_layer[index]["name"][:5] != ms_Parameter_list_for_each_layer[index]["name"][:5]:
        #         pt_Parameter_list_for_each_layer.insert(index, {"name": f"None {pt_Parameter_list_for_each_layer[index]['name']} {index}", "output": None})
        #     index += 1

        # for index in range(maxnum):
        #     print(pt_Parameter_list_for_each_layer[index]["name"])
        #     print(pt_Parameter_list_for_each_layer[index]["output"])
        #     print(ms_Parameter_list_for_each_layer[index]["name"])
        #     print(ms_Parameter_list_for_each_layer[index]["output"])
        #     print()


        # maxnum = len(pt_Parameter_list_for_each_layer) if len(pt_Parameter_list_for_each_layer) < len(ms_Parameter_list_for_each_layer) else len(ms_Parameter_list_for_each_layer)
        # for index in range(maxnum):
        #     print(pt_Parameter_list_for_each_layer[index]["name"])
        #     print(pt_Parameter_list_for_each_layer[index]["output"])
        #     print(ms_Parameter_list_for_each_layer[index]["name"])
        #     print(ms_Parameter_list_for_each_layer[index]["output"])
        #     print()
        #     # if "MetaIBNNorm" not in pt_Parameter_list_for_each_layer[index]["name"] and np.allclose(pt_Parameter_list_for_each_layer[index]["output"], ms_Parameter_list_for_each_layer[index]["output"], atol=1e-5) == False:
        #     #     break
        #     # if pt_Parameter_list_for_each_layer[index]["name"][:5] != ms_Parameter_list_for_each_layer[index]["name"][:5]:
        #     #     break

        self.assertEqual(np.allclose(output_tensor[0].numpy().astype(np.float32), expected_tensor[0].detach().numpy().astype(np.float32), atol=1e-5), True)

    # @unittest.skip("主干网络，不做测试")
    def test_ResNet(self):
        # 初始化模型
        ms_model = test_meta_dynamic_router_resnet_mindspore.build_meta_dynamic_router_resnet_backbone(cfg).set_train(False)
        pt_model = test_meta_dynamic_router_resnet_pytorch.build_meta_dynamic_router_resnet_backbone(cfg).eval()

        # from fastreid.utils.checkpoint import Checkpointer
        # Checkpointer(pt_model).load("/mnt/ssd3/yuming/model_final.pth")

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
            pt_parameter = eval("pt_model.{}.detach().numpy().reshape((-1,))[:20]".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', item))))
            ms_parameter = eval("ms_model.{}.data.asnumpy().reshape((-1,))[:20]".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', Parameter_map[item]))))
            if np.allclose(pt_parameter, ms_parameter, atol=1e-5) != True:
                pt_parameter_all = eval("pt_model.{}.detach().numpy()".format(re.sub(r'\.(\d)\.', '[\\1].', re.sub(r'(\d)\.(\d)', '\\1[\\2]', item))))
                if mindspore_model_dict[Parameter_map[item]].shape == pt_parameter_all.shape:
                    mindspore_model_dict[Parameter_map[item]] = mindspore.Tensor(pt_parameter_all)
                else:
                    print("error: The parameter shapes are different!")

        mindspore.save_checkpoint([{"name": key, "data": mindspore.Tensor(value.numpy())} for key, value in mindspore_model_dict.items()], "/home/yuming/.cache/torch/checkpoints/ACL-DGReID.ckpt")
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
        #         pt_Parameter_list_for_each_layer.append({"name": str(module), "output": output[0].detach().numpy().reshape((-1,))[:20]})
        #         # print("Output shape:", output[0].detach().numpy().reshape((-1,))[:10])
        #     else:
        #         pt_Parameter_list_for_each_layer.append({"name": str(module), "output": output.detach().numpy().reshape((-1,))[:20]})
        #         # print("Output shape:", output.shape)

        # def ms_hook_fn(module, input, output):
        #     if "HyperRouter" in str(module) or "ResNet" in str(module):
        #         ms_Parameter_list_for_each_layer.append({"name": str(module), "output": output[0].asnumpy().reshape((-1,))[:20]})
        #     else:
        #         ms_Parameter_list_for_each_layer.append({"name": str(module), "output": output.asnumpy().reshape((-1,))[:20]})

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
        # maxnum = len(ms_Parameter_list_for_each_layer) if len(pt_Parameter_list_for_each_layer) < len(ms_Parameter_list_for_each_layer) else len(pt_Parameter_list_for_each_layer)
        # index = 0
        # while index < maxnum:
        #     print(pt_Parameter_list_for_each_layer[index]["name"][:5] != ms_Parameter_list_for_each_layer[index]["name"][:5])
        #     if pt_Parameter_list_for_each_layer[index]["name"][:5] != ms_Parameter_list_for_each_layer[index]["name"][:5]:
        #         pt_Parameter_list_for_each_layer.insert(index, {"name": None, "output": None})
        #     index += 1

        # for index in range(maxnum):
        #     print(pt_Parameter_list_for_each_layer[index]["name"])
        #     print(pt_Parameter_list_for_each_layer[index]["output"])
        #     print(ms_Parameter_list_for_each_layer[index]["name"])
        #     print(ms_Parameter_list_for_each_layer[index]["output"])
        #     print()
        #     # if np.allclose(pt_Parameter_list_for_each_layer[index]["output"], ms_Parameter_list_for_each_layer[index]["output"], atol=1e-5) == False:
        #     #     break
    
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

    def test_MetaBNNorm(self):
        # 输入张量的形状应该是(N, C, H, W)，其中 N 是批次大小，C 是通道数，H 是高度，W 是宽度。num_features 应该等于 C，否则会报错。
        num_features = 4
        length = width = height = 2

        input_tensor = (ops.randn(2, num_features, length, length) + 1) / 1000
        # print(input_tensor)
        model = test_ops_mindspore.MetaBNNorm(num_features)
        output_tensor = model(input_tensor)

        input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
        model = test_pytorch.MetaBNNorm(num_features)
        expected_tensor = model(input_tensor)

        # print(f"========= ms_model ==========")
        # print(output_tensor)
        # print(f"========= pt_model ==========")
        # print(expected_tensor)

        self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    def test_MetaINNorm(self):
        num_features = 3
        length = width = height = 1

        # input_tensor = mindspore.Tensor([0.01572487, 0.08070529, 0.08123949, 0.06590732, 0.00857857, 0.04251582, 0.04124985, 0.04972302, 0.00930181, 0.04269622])
        input_tensor = ops.randn(1, num_features, length, length)
        input_tensor[0] = -0.00757481
        # input_tensor[1] = 0.07260463
        model = test_ops_mindspore.MetaINNorm(num_features)
        output_tensor = model(input_tensor)

        input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
        model = test_pytorch.MetaINNorm(num_features)
        expected_tensor = model(input_tensor)

        # print(f"========= ms_model ==========")
        # print(output_tensor)
        # print(f"========= pt_model ==========")
        # print(expected_tensor)

        self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

    def test_MetaIBNNorm(self):
        # num_features 必须是偶数
        num_features = planes = 4
        length = width = height = 2

        input_tensor = ops.randn(1, num_features, length, length)
        model = test_ops_mindspore.MetaIBNNorm(planes)
        output_tensor = model(input_tensor)

        input_tensor = torch.Tensor(input_tensor.numpy().astype(np.float32))
        model = test_pytorch.MetaIBNNorm(planes)
        expected_tensor = model(input_tensor)

        # print(f"========= ms_model ==========")
        # print(output_tensor)
        # print(f"========= pt_model ==========")
        # print(expected_tensor)

        self.assertEqual(np.allclose(output_tensor.numpy().astype(np.float32), expected_tensor.detach().numpy().astype(np.float32), atol=1e-5), True)

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