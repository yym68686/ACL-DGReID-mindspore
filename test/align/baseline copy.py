import os
os.system("clear")
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from torch import nn
from pytorch_fastreid.modeling.backbones import build_backbone
from pytorch_fastreid.modeling.heads import build_heads
from pytorch_fastreid.modeling.losses import *


class Convv(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义模型的层
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)

class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.backbone = Convv()
        self.backbone = build_backbone(cfg)
        # print(self.backbone)
        self.heads = build_heads(cfg)
        # print(self.heads)

if __name__ == "__main__":

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
    cfg.MODEL.HEADS.NUM_CLASSES1 = 600
    cfg.MODEL.HEADS.NUM_CLASSES2 = 600
    cfg.MODEL.HEADS.NUM_CLASSES3 = 600

    pt_model = Baseline(cfg).eval()
    print(pt_model)
