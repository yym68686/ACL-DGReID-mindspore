import numpy as np
import torch
import mindspore as ms
import os
os.system("clear")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import fastreid.modeling.backbones.meta_dynamic_router_resnet as test_meta_dynamic_router_resnet_mindspore
import pytorch_fastreid.modeling.backbones.meta_dynamic_router_resnet as test_meta_dynamic_router_resnet_pytorch

def check_res(pth_path, ckpt_path):
    inp = np.random.uniform(-1, 1, (4, 3, 224, 224)).astype(np.float32)
    # 注意做单元测试时，需要给Cell打训练或推理的标签
    ms_model = test_meta_dynamic_router_resnet_mindspore.build_meta_dynamic_router_resnet_backbone(1)
    pt_model = test_meta_dynamic_router_resnet_pytorch.build_meta_dynamic_router_resnet_backbone(1)

    # ms_model = ms_model50(num_classes=10).set_train(False)
    # pt_model = pt_model50(num_classes=10).eval()
    pt_model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    ms.load_checkpoint(ckpt_path, ms_model)
    print("========= pt_model conv1.weight ==========")
    print(pt_model.conv1.weight.detach().numpy().reshape((-1,))[:10])
    print("========= ms_model conv1.weight ==========")
    print(ms_model.conv1.weight.data.asnumpy().reshape((-1,))[:10])
    pt_res = pt_model(torch.from_numpy(inp))
    ms_res = ms_model(ms.Tensor(inp))
    print("========= pt_model res ==========")
    print(pt_res)
    print("========= ms_model res ==========")
    print(ms_res)
    print("diff", np.max(np.abs(pt_res.detach().numpy() - ms_res.asnumpy())))

# pth_path = "/home/yuming/.cache/torch/checkpoints/resnet50_ibn_a-d9d0bb7b.pth"
pth_path = "/home/yuming/.cache/torch/checkpoints/ACL-DGReID.pth"
ckpt_path = "/home/yuming/.cache/torch/checkpoints/ACL-DGReID2.ckpt"
check_res(pth_path, ckpt_path)
