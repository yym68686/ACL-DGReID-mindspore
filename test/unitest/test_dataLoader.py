import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/fastreid/data")

from fastreid.data.transforms import build_transforms
from fastreid.data.common import CommDataset
from fastreid.data.datasets import DATASET_REGISTRY

from fastreid.config import get_cfg
import mindspore

mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")

cfg = get_cfg()
cfg.merge_from_file("./configs/bagtricks_DR50_mix.yml")

transforms = build_transforms(cfg, is_train=True)

train_items = list()
mapper = dict()
single_set = []
single_sampler = []
num_pids = []




for idx, d in enumerate(cfg.DATASETS.NAMES):
    print(idx,d)
    data = DATASET_REGISTRY.get(d)(root="datasets")
    data.show_train()
    num_pids.append(data.num_train_pids)
    train_items.extend(data.train)
    mapper[d] = idx

train_items = train_items[:50]
train_set = CommDataset(train_items, transforms, relabel=True, mapping=mapper)
train_set.num_classes1 = num_pids[0]
train_set.num_classes2 = num_pids[1]
train_set.num_classes3 = num_pids[2]
# train_set = train_set[0]
print(len(train_set), train_set[0].keys())

train_loader = mindspore.dataset.GeneratorDataset(
    source=train_set,
    column_names=["data"],
    # column_names=["images0", "images", "targets", "camids", "domainids", "img_paths"],
    num_parallel_workers=1,
)

iterator = iter(train_loader)
print(next(iterator))

# dataset = train_loader.batch(2, drop_remainder=True)
# iterator = dataset.create_dict_iterator()
# for data_dict in iterator:
#     print(data_dict)
#     # for name in data_dict.keys():
#     #     print(name, data_dict[name].shape)