import numpy as np
import torch
import mindspore as ms
from collections import OrderedDict

def pytorch2mindspore(pth_file):
    # read pth file
    par_dict = torch.load(pth_file, map_location=torch.device('cpu'))
    params_list = []
    for name in par_dict:
        param_dict = {}
        parameter = par_dict[name]
        param_dict['name'] = name
        param_dict['data'] = ms.Tensor(parameter.numpy())
        params_list.append(param_dict)
    # print(pth_file.split(".pth")[0] + ".ckpt")
    ms.save_checkpoint(params_list, pth_file.split(".pth")[0] + ".ckpt")

# from resnet_ms.src.resnet import resnet50 as ms_resnet50
# from resnet_pytorch.resnet import resnet50 as pt_resnet50

# def check_res(pth_path, ckpt_path):
#     inp = np.random.uniform(-1, 1, (4, 3, 224, 224)).astype(np.float32)
#     # 注意做单元测试时，需要给Cell打训练或推理的标签
#     ms_resnet = ms_resnet50(num_classes=10).set_train(False)
#     pt_resnet = pt_resnet50(num_classes=10).eval()
#     pt_resnet.load_state_dict(torch.load(pth_path, map_location='cpu'))
#     ms.load_checkpoint(ckpt_path, ms_resnet)
#     print("========= pt_resnet conv1.weight ==========")
#     print(pt_resnet.conv1.weight.detach().numpy().reshape((-1,))[:10])
#     print("========= ms_resnet conv1.weight ==========")
#     print(ms_resnet.conv1.weight.data.asnumpy().reshape((-1,))[:10])
#     pt_res = pt_resnet(torch.from_numpy(inp))
#     ms_res = ms_resnet(ms.Tensor(inp))
#     print("========= pt_resnet res ==========")
#     print(pt_res)
#     print("========= ms_resnet res ==========")
#     print(ms_res)
#     print("diff", np.max(np.abs(pt_res.detach().numpy() - ms_res.asnumpy())))

# pth_path = "resnet.pth"
# ckpt_path = "resnet50.ckpt"
# check_res(pth_path, ckpt_path)

def param_convert(ms_params, pt_params, ckpt_path = "./ACL-DGReID2.ckpt"):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}
    new_params_list = []
    Parameter_map = OrderedDict()
    Parameter_all = OrderedDict()
    # print(ms_params.keys())
    for ms_param in ms_params.keys():
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数
        if "bn" in ms_param or "downsample.1" in ms_param or "norm" in ms_param or "map" in ms_param or "moving_mean" in ms_param or "moving_variance" in ms_param or "gamma" in ms_param or "beta" in ms_param:
            ms_param_item = ms_param.split(".")
            pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            pt_param = ".".join(pt_param_item)
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[pt_param]
                Parameter_map[pt_param] = ms_param
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value.cpu().numpy().astype(np.float32))})
            else:
                # print(ms_param, "not match in pt_params")
                pass
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表
            # print(ms_param, ms_param in pt_params, pt_params[ms_param].shape == ms_params[ms_param].shape)
            if ms_param in pt_params and pt_params[ms_param].shape == ms_params[ms_param].shape:
                ms_value = pt_params[ms_param]
                # print(type(ms_value))
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value.cpu().numpy().astype(np.float32))})
                Parameter_map[ms_param] = ms_param
            else:
                # print(ms_param, "not match in pt_params")
                pass
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)
    return Parameter_map
    # print(Parameter_map)

def WeightTransfer():
    Parameter_map = param_convert()


if __name__ == '__main__':
    pytorch2mindspore("/home/yuming/.cache/torch/checkpoints/resnet50_ibn_a-d9d0bb7b.pth")