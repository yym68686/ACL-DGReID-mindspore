import mindspore
from mindspore.communication import init, get_rank, get_group_size
from fastreid.utils.events import EventStorage
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
import os
import re
import copy
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from collections import OrderedDict
from enum import Enum
import logging
logger = logging.getLogger(__name__)



from fastreid.data.datasets import DATASET_REGISTRY
from mindspore.dataset import Dataset
from PIL import Image, ImageOps
import numpy as np
import argparse

from fastreid.data.transforms import build_transforms

def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"
    Returns:
        image (np.ndarray): an HWC image
    """
    with open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)

        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]

        # handle grayscale mixed in RGB images
        elif len(image.shape) == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        image = Image.fromarray(image)

        return image

class CommDataset(Dataset):
    """Image Person ReID Dataset"""
    #CHANGE Add domain id

    def __init__(self, img_items, transform=None, relabel=True, mapping=None, offset=0):
        super().__init__()
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.mapping = mapping

        assert self.mapping is not None, 'mapping must be initialized!!!'

        if isinstance(self.mapping, dict):
            pid_set = [set() for i in range(len(self.mapping))]
            cam_set = [set() for i in range(len(self.mapping))]
            for i in img_items:
                domain_id = self.mapping[i[1].split("_")[0]]
                pid_set[domain_id].add(i[1])
                cam_set[domain_id].add(i[2])

            self.pids = [] 
            self.cams = [] 
            for temp_pid, temp_cam in zip(pid_set, cam_set):
                self.pids += sorted(list(temp_pid))
                self.cams += sorted(list(temp_cam))
        else:
            pid_set = set()
            cam_set = set()
            for i in img_items:
                pid_set.add(i[1])
                cam_set.add(i[2])

            self.pids = sorted(list(pid_set))
            self.cams = sorted(list(cam_set))
        
        if relabel:
            self.pid_dict = dict([(p, i+offset) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        # QUES
        if self.transform is not None:
            img0 = self.transform[0](img)
            img = self.transform[1](img)
        if self.mapping and isinstance(self.mapping, dict):
            domain_id = self.mapping[pid.split("_")[0]]
        else:
            domain_id = self.mapping
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        if isinstance(img0, tuple):
            img0 = img0[0]
        if isinstance(img, tuple):
            img = img[0]
        # return img, pid, camid, domain_id, img_path
        return img0, img, pid, camid, domain_id, img_path
        # return {
        #     "images0": img0,
        #     "images": img,
        #     "targets": pid,
        #     "camids": camid,
        #     "domainids": domain_id,
        #     "img_paths": img_path,
        # }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
    
    def __call__(self):
        # print("111")
        pass

def build_reid_train_loader(cfg):
    transforms = build_transforms(cfg, is_train=True)
    train_items = list()
    mapper = dict()
    num_pids = []
    for idx, d in enumerate(cfg.DATASETS.NAMES):
        data = DATASET_REGISTRY.get(d)(root="datasets")
        data.show_train()
        num_pids.append(data.num_train_pids)
        train_items.extend(data.train)
        mapper[d] = idx

    train_set = CommDataset(train_items, transforms, relabel=True, mapping=mapper)
    train_set.num_classes1 = num_pids[0]
    train_set.num_classes2 = num_pids[1]
    train_set.num_classes3 = num_pids[2]
    # print(train_set[0])

    train_loader = mindspore.dataset.GeneratorDataset(
        source=train_set,
        # column_names=["images" ,"targets" ,"camids" ,"domainids" ,"img_paths"],
        column_names=["images0" ,"images" ,"targets" ,"camids" ,"domainids" ,"img_paths"],
        num_parallel_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    # import mindspore.dataset.transforms as T
    # from fastreid.data.transforms import AutoAugment


    # train_loader = train_loader.map(operations=T.RandomApply([AutoAugment()], prob=cfg.INPUT.AUTOAUG.PROB),
    #                           input_columns=["images"],
    #                           output_columns=["images0"],
    #                           num_parallel_workers=cfg.DATALOADER.NUM_WORKERS)
    # train_loader = train_loader.map(operations=[transforms[0]],
    #                           input_columns=["images" ,"targets" ,"camids" ,"domainids" ,"img_paths"],
    #                           output_columns=["images0" ,"images" ,"targets" ,"camids" ,"domainids" ,"img_paths"],
    #                           num_parallel_workers=cfg.DATALOADER.NUM_WORKERS)
    
    # train_loader = train_loader.map(operations=[transforms[1]],
    #                           input_columns=["images"],
    #                           output_columns=["images"],
    #                         #   input_columns=["images0" ,"images" ,"targets" ,"camids" ,"domainids" ,"img_paths"],
    #                         #   output_columns=["images0" ,"images" ,"targets" ,"camids" ,"domainids" ,"img_paths"],
    #                           num_parallel_workers=cfg.DATALOADER.NUM_WORKERS)

    return train_loader, [train_set.num_classes1, train_set.num_classes2, train_set.num_classes3]

def _generate_optimizer_class_with_freeze_layer(
        optimizer: Type[mindspore.nn.Optimizer],
        # optimizer: Type[torch.optim.Optimizer],
        *,
        freeze_iters: int = 0,
) -> Type[mindspore.nn.Optimizer]:
# ) -> Type[torch.optim.Optimizer]:
    assert freeze_iters > 0, "No layers need to be frozen or freeze iterations is 0"

    cnt = 0
    # @torch.no_grad()
    def optimizer_wfl_step(self, closure=None):
        nonlocal cnt
        if cnt < freeze_iters:
            cnt += 1
            param_ref = []
            grad_ref = []
            for group in self.param_groups:
                if group["freeze_status"] == "freeze":
                    for p in group["params"]:
                        if p.grad is not None:
                            param_ref.append(p)
                            grad_ref.append(p.grad)
                            p.grad = None

            optimizer.step(self, closure)
            for p, g in zip(param_ref, grad_ref):
                p.grad = g
        else:
            optimizer.step(self, closure)

    OptimizerWithFreezeLayer = type(
        optimizer.__name__ + "WithFreezeLayer",
        (optimizer,),
        {"step": optimizer_wfl_step},
    )
    return OptimizerWithFreezeLayer


def get_default_optimizer_params(
        model: mindspore.nn.Cell,
        # model: torch.nn.Module,
        base_lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        weight_decay_norm: Optional[float] = None,
        bias_lr_factor: Optional[float] = 1.0,
        heads_lr_factor: Optional[float] = 1.0,
        weight_decay_bias: Optional[float] = None,
        overrides: Optional[Dict[str, Dict[str, float]]] = None,
        freeze_layers: Optional[list] = [],
        flag=None
):
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.
    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        heads_lr_factor: multiplier of lr for model.head parameters.
        weight_decay_bias: override weight decay for bias parameters
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.
        freeze_layers: layer names for freezing.
    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.
    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        # NOTE: unlike Detectron v1, we now by default make bias hyperparameters
        # exactly the same as regular weights.
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides

    layer_names_pattern = [re.compile(name) for name in freeze_layers]

    norm_module_types = (
        mindspore.nn.BatchNorm1d,
        mindspore.nn.BatchNorm2d,
        mindspore.nn.BatchNorm3d,
        mindspore.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        mindspore.nn.GroupNorm,
        mindspore.nn.InstanceNorm1d,
        mindspore.nn.InstanceNorm2d,
        mindspore.nn.InstanceNorm3d,
        mindspore.nn.LayerNorm,
        # mindspore.nn.LocalResponseNorm,
    )
    # norm_module_types = (
    #     torch.nn.BatchNorm1d,
    #     torch.nn.BatchNorm2d,
    #     torch.nn.BatchNorm3d,
    #     torch.nn.SyncBatchNorm,
    #     # NaiveSyncBatchNorm inherits from BatchNorm2d
    #     torch.nn.GroupNorm,
    #     torch.nn.InstanceNorm1d,
    #     torch.nn.InstanceNorm2d,
    #     torch.nn.InstanceNorm3d,
    #     torch.nn.LayerNorm,
    #     torch.nn.LocalResponseNorm,
    # )
    params: List[Dict[str, Any]] = []
    memo: Set[mindspore.Parameter] = set()
    # memo: Set[torch.nn.parameter.Parameter] = set()

    # for module_name, module in model.named_modules():
        # for module_param_name, value in module.named_parameters(recurse=False):
    for module_name, module in model.cells_and_names():
        for item in module.get_parameters():
            module_param_name = item.name
            value = item.value()

            value = mindspore.Tensor(value, mindspore.float32)
            value = mindspore.Parameter(value)
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            hyperparams.update(overrides.get(module_param_name, {}))
            if module_name.split('.')[0] == "heads" and (heads_lr_factor is not None and heads_lr_factor != 1.0):
                hyperparams["lr"] = hyperparams.get("lr", base_lr) * heads_lr_factor
            #CHANGE Imporve the learning rate for sub module
            # if 'bn4' in module_name:
            #     hyperparams["lr"] = hyperparams.get("lr", base_lr) * 3
            if 'adaptor_sub' in module_name:
                hyperparams["lr"] = hyperparams.get("lr", base_lr) * .5
            # if 'dynamic' in module_name:
            #     hyperparams["lr"] = hyperparams.get("lr", base_lr) * 2.0

            name = module_name + '.' + module_param_name
            freeze_status = "normal"
            # Search freeze layer names, it must match from beginning, so use `match` not `search`
            for pattern in layer_names_pattern:
                if pattern.match(name) is not None:
                    freeze_status = "freeze"
                    break
            # if flag == 'meta' and ('router' not in module_name or 'adaptor' not in module_name):
            #     freeze_status = "freeze"
            if flag == 'meta' and ('router' not in module_name or 'fc_classifier' in module_name):
                freeze_status = "freeze"
            # if flag is None and 'router' in module_name:
            #     freeze_status = "freeze"

            params.append({"params": [value], **hyperparams})
            # params.append({"freeze_status": freeze_status, "params": [value], **hyperparams})
    return params

def maybe_add_freeze_layer(
        cfg, optimizer: Type[mindspore.nn.Optimizer]
        # cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[mindspore.nn.Optimizer]:
    if len(cfg.MODEL.FREEZE_LAYERS) == 0 or cfg.SOLVER.FREEZE_ITERS <= 0:
        return optimizer

    # if isinstance(optimizer, torch.optim.Optimizer):
    if isinstance(optimizer, mindspore.nn.Optimizer):
        optimizer_type = type(optimizer)
    else:
        # assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        assert issubclass(optimizer, mindspore.nn.Optimizer), optimizer
        optimizer_type = optimizer

    OptimizerWithFreezeLayer = _generate_optimizer_class_with_freeze_layer(
        optimizer_type,
        freeze_iters=cfg.SOLVER.FREEZE_ITERS
    )
    # if isinstance(optimizer, torch.optim.Optimizer):
    print(optimizer, isinstance(optimizer, mindspore.nn.Optimizer))
    if isinstance(optimizer, mindspore.nn.Optimizer):
        optimizer.__class__ = OptimizerWithFreezeLayer  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithFreezeLayer

class ContiguousParams:

    def __init__(self, parameters):
        # Create a list of the parameters to prevent emptying an iterator.
        self._parameters = parameters
        self._param_buffer = []
        self._grad_buffer = []
        self._group_dict = OrderedDict()
        self._name_buffer = []
        self._init_buffers()
        # Store the data pointers for each parameter into the buffer. These
        # can be used to check if an operation overwrites the gradient/data
        # tensor (invalidating the assumption of a contiguous buffer).
        self.data_pointers = []
        self.grad_pointers = []
        self.make_params_contiguous()

    def _init_buffers(self):
        dtype = self._parameters[0]["params"][0].dtype
        device = self._parameters[0]["params"][0].device
        if not all(p["params"][0].dtype == dtype for p in self._parameters):
            raise ValueError("All parameters must be of the same dtype.")
        if not all(p["params"][0].device == device for p in self._parameters):
            raise ValueError("All parameters must be on the same device.")

        # Group parameters by lr and weight decay
        for param_dict in self._parameters:
            freeze_status = param_dict["freeze_status"]
            param_key = freeze_status + '_' + str(param_dict["lr"]) + '_' + str(param_dict["weight_decay"])
            if param_key not in self._group_dict:
                self._group_dict[param_key] = []
            self._group_dict[param_key].append(param_dict)

        for key, params in self._group_dict.items():
            size = sum(p["params"][0].numel() for p in params)
            self._param_buffer.append(torch.zeros(size, dtype=dtype, device=device))
            self._grad_buffer.append(torch.zeros(size, dtype=dtype, device=device))
            self._name_buffer.append(key)

    def make_params_contiguous(self):
        """Create a buffer to hold all params and update the params to be views of the buffer.
        Args:
            parameters: An iterable of parameters.
        """
        for i, params in enumerate(self._group_dict.values()):
            index = 0
            for param_dict in params:
                p = param_dict["params"][0]
                size = p.numel()
                self._param_buffer[i][index:index + size] = p.data.view(-1)
                p.data = self._param_buffer[i][index:index + size].view(p.data.shape)
                p.grad = self._grad_buffer[i][index:index + size].view(p.data.shape)
                self.data_pointers.append(p.data.data_ptr)
                self.grad_pointers.append(p.grad.data.data_ptr)
                index += size
            # Bend the param_buffer to use grad_buffer to track its gradients.
            self._param_buffer[i].grad = self._grad_buffer[i]

    def contiguous(self):
        """Return all parameters as one contiguous buffer."""
        return [{
            "freeze_status": self._name_buffer[i].split('_')[0],
            "params": self._param_buffer[i],
            "lr": float(self._name_buffer[i].split('_')[1]),
            "weight_decay": float(self._name_buffer[i].split('_')[2]),
        } for i in range(len(self._param_buffer))]

    def original(self):
        """Return the non-flattened parameters."""
        return self._parameters

    def buffer_is_valid(self):
        """Verify that all parameters and gradients still use the buffer."""
        i = 0
        for params in self._group_dict.values():
            for param_dict in params:
                p = param_dict["params"][0]
                data_ptr = self.data_pointers[i]
                grad_ptr = self.grad_pointers[i]
                if (p.data.data_ptr() != data_ptr()) or (p.grad.data.data_ptr() != grad_ptr()):
                    return False
                i += 1
        return True

    def assert_buffer_is_valid(self):
        if not self.buffer_is_valid():
            raise ValueError(
                "The data or gradient buffer has been invalidated. Please make "
                "sure to use inplace operations only when updating parameters "
                "or gradients.")

_GradientClipperInput = Union[mindspore.Tensor, Iterable[mindspore.Tensor]]
# _GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"

def _create_gradient_clipper(cfg):
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]

def _generate_optimizer_class_with_gradient_clipping(
        optimizer: Type[mindspore.nn.Optimizer],
        # optimizer: Type[torch.optim.Optimizer],
        *,
        per_param_clipper: Optional[_GradientClipper] = None,
        global_clipper: Optional[_GradientClipper] = None,
) -> Type[mindspore.nn.Optimizer]:
# ) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
            per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    # @torch.no_grad()
    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        optimizer.step(self, closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip

def maybe_add_gradient_clipping(
        cfg, optimizer: Type[mindspore.nn.Optimizer]
        # cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[mindspore.nn.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.
    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer
    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip

def build_optimizer(cfg, model, contiguous=False, flag=None):
    solver_opt = cfg.SOLVER.OPT
    if flag == 'meta':
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            heads_lr_factor=cfg.SOLVER.HEADS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            freeze_layers=cfg.MODEL.FREEZE_LAYERS if cfg.SOLVER.FREEZE_ITERS > 0 else [],
            flag='meta'
        )
        if contiguous:
            params = ContiguousParams(params)
        if solver_opt == "SGD":
            return maybe_add_freeze_layer(
                cfg,
                maybe_add_gradient_clipping(cfg, mindspore.nn.optim_ex.SGD)
                # maybe_add_gradient_clipping(cfg, torch.optim.SGD)
            )(
                # params.contiguous() if contiguous else params,
                params,
                # learning_rate=0.01,
                lr=0.01,
                momentum=0,
                nesterov=False,
            ), params
        else:
            return maybe_add_freeze_layer(
                cfg,
                maybe_add_gradient_clipping(cfg, getattr(torch.optim, solver_opt))
            )(params.contiguous() if contiguous else params), params

    else:
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            heads_lr_factor=cfg.SOLVER.HEADS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            freeze_layers=cfg.MODEL.FREEZE_LAYERS if cfg.SOLVER.FREEZE_ITERS > 0 else [],
        )
        # print(type(params), params)
        if contiguous:
            params = ContiguousParams(params)
        if solver_opt == "SGD":
            return maybe_add_freeze_layer(
                cfg,
                maybe_add_gradient_clipping(cfg, mindspore.nn.SGD)
                # maybe_add_gradient_clipping(cfg, mindspore.nn.optim_ex.SGD)
                # maybe_add_gradient_clipping(cfg, torch.optim.SGD)
            )(
                params,
                # params.contiguous() if contiguous else params,
                learning_rate=0.01,
                # lr=0.01,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            ), params
        else:
            return maybe_add_freeze_layer(
                cfg,
                maybe_add_gradient_clipping(cfg, getattr(torch.optim, solver_opt))
            )(params.contiguous() if contiguous else params), params

def custom_per_batch_map(data, BatchInfo):
    # 自定义处理逻辑，将字典对象转换为适当的NumPy数组
    # print(type(data['images0']))
    # data_list = []
    images0_tensor = None
    images_tensor = None
    targets_tensor = None
    camids_tensor = None
    domainids_tensor = None
    img_paths_tensor = None
    for item in data:
        images0_tensor = mindspore.ops.stack((item["images0"], images0_tensor), 0)
        images_tensor = mindspore.ops.stack((item["images"], images_tensor), 0)
        targets_tensor = mindspore.ops.stack((item["targets"], targets_tensor), 0)
        camids_tensor = mindspore.ops.stack((item["camids"], camids_tensor), 0)
        domainids_tensor = mindspore.ops.stack((item["domainids"], domainids_tensor), 0)
        img_paths_tensor = mindspore.ops.stack((item["img_paths"], img_paths_tensor), 0)
    #     data_list.append({
    #     "images0": item["images0"],
    #     "images": item["images"],
    #     "targets": item["targets"],
    #     "camids": item["camids"],
    #     "domainids": item["domainids"],
    #     "img_paths": item["img_paths"],
    # })
    # return data_list  # 返回转换后的数组
    return images0_tensor, images_tensor, targets_tensor, camids_tensor, domainids_tensor, img_paths_tensor

os.system("clear")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU", device_id=3)
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")
# init("nccl")
# mindspore.reset_auto_parallel_context()
# device_num = get_group_size()
# # print(device_num)
# mindspore.set_auto_parallel_context(device_num=device_num, parallel_mode=mindspore.ParallelMode.DATA_PARALLEL, parameter_broadcast=True, gradients_mean=True)

# 参数解析
parser = argparse.ArgumentParser(description="fastreid Training")
parser.add_argument("--config-file", default="./configs/bagtricks_DR50_mix.yml", metavar="FILE", help="path to config file")
parser.add_argument("--num-gpus", type=int, default=4, help="number of gpus *per machine*")
args = parser.parse_args()
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
default_setup(cfg, args)

# dataloader
data_loader, num_classes = build_reid_train_loader(cfg)

cfg.MODEL.HEADS.NUM_CLASSES1 = num_classes[0]
cfg.MODEL.HEADS.NUM_CLASSES2 = num_classes[1]
cfg.MODEL.HEADS.NUM_CLASSES3 = num_classes[2]

from fastreid.modeling.meta_arch.build import build_model
model = build_model(cfg)
model.set_train(True)
# mindspore.load_checkpoint("/home/yuming/.cache/torch/checkpoints/ACL-DGReID.ckpt", model)

optimizer = build_optimizer(cfg, model)[0]
# print(type(optimizer), type(param_wrapper), optimizer, param_wrapper)
# exit(0)

# dataset = data_loader.batch(cfg.SOLVER.IMS_PER_BATCH, drop_remainder=True)
# dataset = data_loader.batch(cfg.SOLVER.IMS_PER_BATCH, True, output_columns=["images" ,"targets" ,"camids" ,"domainids" ,"img_paths"])
dataset = data_loader.batch(cfg.SOLVER.IMS_PER_BATCH, True, output_columns=["images0" ,"images" ,"targets" ,"camids" ,"domainids" ,"img_paths"])

# preprocess_image
pixel_mean = mindspore.Tensor([0.485*255, 0.456*255, 0.406*255]).view(1, -1, 1, 1)
pixel_std = mindspore.Tensor([0.229*255, 0.224*255, 0.225*255]).view(1, -1, 1, 1)



max_epoch = cfg.SOLVER.MAX_EPOCH
start_epoch = 0
iters_per_epoch = len(data_loader) // cfg.SOLVER.IMS_PER_BATCH
iter_ = start_iter = start_epoch * iters_per_epoch
print(f"start_iter: {start_iter}, batch size: {cfg.SOLVER.IMS_PER_BATCH}, iters_per_epoch: {iters_per_epoch}, max_epoch: {max_epoch}")
with EventStorage(start_iter) as storage:
    # try:
        for epoch in range(start_epoch, max_epoch):
            dataset_iter = iter(dataset)
            for _ in range(iters_per_epoch):
                # images, targets, camids, domainids, img_paths = next(dataset_iter)
                images0, images, targets, camids, domainids, img_paths = next(dataset_iter)
                images = images.sub(pixel_mean).div(pixel_std)
                weights = optimizer.parameters
                opt = -1
                # loss_dict, inputs_gradient = mindspore.value_and_grad(model, grad_position=None, weights=weights, has_aux=False)(images, targets, camids, domainids, img_paths, epoch, opt)
                (loss_cls, loss_center, loss_triplet, loss_circle, loss_cosface, loss_triplet_add, loss_triplet_mtrain, loss_stc, loss_triplet_mtest, loss_domain_intra, loss_domain_inter, loss_Center), inputs_gradient = mindspore.value_and_grad(model, grad_position=None, weights=weights, has_aux=False)(images, targets, domainids, epoch, opt)
                # loss_dict, inputs_gradient = mindspore.value_and_grad(model, grad_position=None, weights=weights, has_aux=False)(images0, images, targets, camids, domainids, img_paths, epoch, opt)
                # print(loss_cls, loss_center, loss_triplet, loss_circle, loss_cosface, loss_triplet_add, loss_triplet_mtrain, loss_stc, loss_triplet_mtest, loss_domain_intra, loss_domain_inter, loss_Center)
                optimizer(inputs_gradient)
                # total_loss = loss_dict.sum()
                total_loss = loss_cls + loss_center + loss_triplet + loss_circle + loss_cosface + loss_triplet_add + loss_triplet_mtrain + loss_stc + loss_triplet_mtest + loss_domain_intra + loss_domain_inter + loss_Center
                print(f"epoch: {epoch}, iters: {_}, loss: {total_loss}")
                iter_ += 1
            mindspore.save_checkpoint(model, f"./result/epoch_{epoch}_loss_{total_loss}.ckpt")
    # except Exception:
    #     logger.exception("Exception during training:")
    #     raise