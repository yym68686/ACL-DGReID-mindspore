# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file ./configs/bagtricks_DR50_mix.yml --num-gpus 1

import logging
import random
import copy
import re
import math
import weakref

import numpy as np
from typing import Dict, Optional
from PIL import Image, ImageOps, ImageEnhance
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

import os
import sys
sys.path.append('.')
import mindspore
import mindspore.dataset.transforms as T
import mindspore.dataset.vision as TV
from mindspore.dataset import Dataset


from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg

def auto_scale_hyperparams(cfg, num_classes1, num_classes2, num_classes3):
    r"""
    This is used for auto-computation actual training iterations,
    because some hyper-param, such as MAX_ITER, means training epochs rather than iters,
    so we need to convert specific hyper-param to training iterations.
    """
    cfg = cfg.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()

    # If you don't hard-code the number of classes, it will compute the number automatically
    if cfg.MODEL.HEADS.NUM_CLASSES1 == 0:
        output_dir = cfg.OUTPUT_DIR
        cfg.MODEL.HEADS.NUM_CLASSES1 = num_classes1
        cfg.MODEL.HEADS.NUM_CLASSES2 = num_classes2
        cfg.MODEL.HEADS.NUM_CLASSES3 = num_classes3
        logger = logging.getLogger(__name__)
        logger.info(f"Auto-scaling the num_classes={cfg.MODEL.HEADS.NUM_CLASSES1}")
        logger.info(f"Auto-scaling the num_classes={cfg.MODEL.HEADS.NUM_CLASSES2}")
        logger.info(f"Auto-scaling the num_classes={cfg.MODEL.HEADS.NUM_CLASSES3}")

        # Update the saved config file to make the number of classes valid
        # if comm.is_main_process() and output_dir:
        if output_dir:
            # Note: some of our scripts may expect the existence of
            # config.yaml in output directory
            path = os.path.join(output_dir, "config.yaml")
            with open(path, "w") as f:
                f.write(cfg.dump())

    if frozen: cfg.freeze()

    return cfg

_HPARAMS_DEFAULT = dict(
    translate_const=57,
    img_mean=_FILL,
)

def auto_augment_policy(name="original"):
    hparams = _HPARAMS_DEFAULT
    if name == 'original':
        return auto_augment_policy_original(hparams)
    elif name == 'originalr':
        return auto_augment_policy_originalr(hparams)
    elif name == 'v0':
        return auto_augment_policy_v0(hparams)
    elif name == 'v0r':
        return auto_augment_policy_v0r(hparams)
    else:
        assert False, 'Unknown AA policy (%s)' % name

class AutoAugment:

    def __init__(self):
        self.policy = auto_augment_policy()

    def __call__(self, img):
        sub_policy = random.choice(self.policy)
        for op in sub_policy:
            img = op(img)
        return img

def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if isinstance(pic, np.ndarray):
        assert len(pic.shape) in (2, 3)
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = mindspore.Tensor.from_numpy(pic.transpose((2, 0, 1)))
        # img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, mindspore.byte):
        # if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

    # print("type(img)", type(img))
    # handle PIL Image
    if pic.mode == 'I':
        img = mindspore.Tensor.from_numpy(np.array(pic, np.int32, copy=False))
        # img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = mindspore.Tensor.from_numpy(np.array(pic, np.int16, copy=False))
        # img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = mindspore.Tensor.from_numpy(np.array(pic, np.float32, copy=False))
        # img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * mindspore.Tensor.from_numpy(np.array(pic, np.uint8, copy=False))
        # img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = mindspore.Tensor(np.array(pic))
        # img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    # print("type(img)", type(img))
    # print("pic.size[1]", pic.size[1])
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    # print("type(img)", type(img))
    img = img.swapaxes(0, 1).swapaxes(0, 2)
    # img = img.transpose(0, 1).transpose(0, 2).contiguous()
    # print("type(img)", type(img))
    if img.dtype == mindspore.uint8:
    # if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, *args):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, *args):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level, *args):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level, *args):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level, *args):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[1] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level, *args):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level, *args):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level, *args):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level, *args):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1, prob_flip_leftright=0.5,
                 ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = mindspore.ops.flip(patch, dims=[2])
            # patch = torch.flip(patch, dims=[2])
        return patch

    def __call__(self, img):
        _, H, W = img.size()  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img[..., y1:y1 + h, x1:x1 + w]
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        _, patchH, patchW = patch.size()
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img[..., y1:y1 + patchH, x1:x1 + patchW] = patch

        return img

class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    """

    def __init__(self, prob=0.5, aug_prob_coeff=0.1, mixture_width=3, mixture_depth=1, aug_severity=1):
        """
        Args:
            prob: Probability of taking augmix
            aug_prob_coeff: Probability distribution coefficients.
            mixture_width: Number of augmentation chains to mix per augmented example.
            mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
            aug_severity: Severity of underlying augmentation operators (between 1 to 10).
        """
        # fmt: off
        self.prob           = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width  = mixture_width
        self.mixture_depth  = mixture_depth
        self.aug_severity   = aug_severity
        self.augmentations  = augmentations
        # fmt: on

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.

        Returns:
          mixed: Augmented and mixed image.
        """
        if random.random() > self.prob:
            # Avoid the warning: the given NumPy array is not writeable
            return np.asarray(image).copy()

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = np.zeros([image.size[1], image.size[0], 3])
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * image + m * mix
        return mixed.astype(np.uint8)

def build_transforms(cfg, is_train=True):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # crop
        do_crop = cfg.INPUT.CROP.ENABLED
        crop_size = cfg.INPUT.CROP.SIZE
        crop_scale = cfg.INPUT.CROP.SCALE
        crop_ratio = cfg.INPUT.CROP.RATIO

        # augmix augmentation
        do_augmix = cfg.INPUT.AUGMIX.ENABLED
        augmix_prob = cfg.INPUT.AUGMIX.PROB

        # auto augmentation
        do_autoaug = cfg.INPUT.AUTOAUG.ENABLED
        autoaug_prob = cfg.INPUT.AUTOAUG.PROB

        # horizontal filp
        do_flip = cfg.INPUT.FLIP.ENABLED
        flip_prob = cfg.INPUT.FLIP.PROB

        # padding
        do_pad = cfg.INPUT.PADDING.ENABLED
        padding_size = cfg.INPUT.PADDING.SIZE
        padding_mode = eval("TV." + cfg.INPUT.PADDING.MODE)

        # color jitter
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE

        # random affine
        do_affine = cfg.INPUT.AFFINE.ENABLED

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_value = cfg.INPUT.REA.VALUE

        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        if do_autoaug:
            res.append(T.RandomApply([AutoAugment()], prob=autoaug_prob))
            # res.append(T.RandomApply([AutoAugment()], p=autoaug_prob))

        if size_train[0] > 0:
            res.append(TV.Resize(size_train[0] if len(size_train) == 1 else size_train, interpolation=TV.Inter.BICUBIC))
            # res.append(T.Resize(size_train[0] if len(size_train) == 1 else size_train, interpolation=3))

        if do_crop:
            res.append(TV.RandomResizedCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size,
                                           interpolation=3,
                                           scale=crop_scale, ratio=crop_ratio))
            # res.append(T.RandomResizedCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size,
            #                                interpolation=3,
            #                                scale=crop_scale, ratio=crop_ratio))
        if do_pad:
            res.extend([TV.Pad(padding_size, padding_mode=padding_mode),
                        TV.RandomCrop(size_train[0] if len(size_train) == 1 else size_train)])
            # res.extend([T.Pad(padding_size, padding_mode=padding_mode),
            #             T.RandomCrop(size_train[0] if len(size_train) == 1 else size_train)])
        if do_flip:
            res.append(TV.RandomHorizontalFlip(prob=flip_prob))
            # res.append(T.RandomHorizontalFlip(p=flip_prob))

        res1 = copy.deepcopy(res)
        res1.append(T.RandomApply([TV.RandomColorAdjust(cj_brightness, cj_contrast, cj_saturation, cj_hue)], prob=cj_prob))
        # res1.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_cj:
            res.append(T.RandomApply([TV.RandomColorAdjust(cj_brightness, cj_contrast, cj_saturation, cj_hue)], prob=cj_prob))
            # res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_affine:
            res.append(TV.RandomAffine(degrees=10, translate=None, scale=[0.9, 1.1], shear=0.1, resample=False,
                                      fill_value=0))
            # res.append(T.RandomAffine(degrees=10, translate=None, scale=[0.9, 1.1], shear=0.1, resample=False,
            #                           fillcolor=0))
        if do_augmix:
            res.append(AugMix(prob=augmix_prob))
        res1.append(ToTensor())
        res.append(ToTensor())
        if do_rea:
            res1.append(TV.RandomErasing(prob=rea_prob, value=rea_value))
            res.append(TV.RandomErasing(prob=rea_prob, value=rea_value))
            # res1.append(T.RandomErasing(p=rea_prob, value=rea_value))
            # res.append(T.RandomErasing(p=rea_prob, value=rea_value))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))

        return [T.Compose(res), T.Compose(res1)]
    else:
        size_test = cfg.INPUT.SIZE_TEST
        do_crop = cfg.INPUT.CROP.ENABLED
        crop_size = cfg.INPUT.CROP.SIZE

        if size_test[0] > 0:
            res.append(TV.Resize(size_test[0] if len(size_test) == 1 else size_test, interpolation=TV.Inter.BICUBIC))
            # res.append(T.Resize(size_test[0] if len(size_test) == 1 else size_test, interpolation=3))
        if do_crop:
            res.append(TV.CenterCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size))
            # res.append(T.CenterCrop(size=crop_size[0] if len(crop_size) == 1 else crop_size))
        res.append(ToTensor())
        
        return [T.Compose(res), T.Compose(res)]

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
    # with PathManager.open(file_name, "rb") as f:
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
        # return img0, img, pid, camid, domain_id, img_path
        return {
            "images0": img0,
            "images": img,
            "targets": pid,
            "camids": camid,
            "domainids": domain_id,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
    
    def __call__(self):
        print("111")

class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        assert (
                name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: object = None) -> Optional[object]:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name
                )
            )
        return ret

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

def build_train_loader(cfg):
    logger = logging.getLogger(__name__)
    logger.info("Prepare training set")
    return build_reid_train_loader(_train_loader_from_config(cfg, combineall=cfg.DATASETS.COMBINEALL))

_root = os.getenv("FASTREID_DATASETS", "datasets")

def _train_loader_from_config(cfg, *, train_set=None, transforms=None, sampler=None, **kwargs):
    if transforms is None:
        transforms = build_transforms(cfg, is_train=True)

    if train_set is None:
        train_items = list()
        mapper = dict()
        single_set = []
        single_sampler = []
        num_pids = []
        #CHANGE Add domain id
        for idx, d in enumerate(cfg.DATASETS.NAMES):
            data = DATASET_REGISTRY.get(d)(root=_root, **kwargs)
            if comm.is_main_process():
                data.show_train()
            else:
                data.show_train(False)
            single_set.append(CommDataset(data.train, transforms, relabel=True, mapping=idx, offset=sum(num_pids)))
            num_pids.append(data.num_train_pids)
            train_items.extend(data.train)
            # print("data.train", type(data.train), data.train)
            mapper[d] = idx
        print(mapper, num_pids)

        train_set = CommDataset(train_items, transforms, relabel=True, mapping=mapper)
        for temp_set in single_set:
            temp_set.pid_dict = train_set.pid_dict
        train_set.num_classes1 = num_pids[0]
        train_set.num_classes2 = num_pids[1]
        train_set.num_classes3 = num_pids[2]
        for i in range(len(single_set)):
            single_set[i].num_classes1 = num_pids[0]
            single_set[i].num_classes2 = num_pids[1]
            single_set[i].num_classes3 = num_pids[2]

    # if sampler is None:
    #     sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    #     num_instance = cfg.DATALOADER.NUM_INSTANCE
    #     mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    #     logger = logging.getLogger(__name__)
    #     logger.info("Using training sampler {}".format(sampler_name))
    #     if sampler_name == "TrainingSampler":
    #         sampler = samplers.TrainingSampler(len(train_set))
    #     elif sampler_name == "NaiveIdentitySampler":
    #         sampler = samplers.NaiveIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
    #         for i in range(len(single_set)):
    #             single_sampler.append(samplers.NaiveIdentitySampler(single_set[i].img_items, mini_batch_size // 2, num_instance))
    #     elif sampler_name == "BalancedIdentitySampler":
    #         sampler = samplers.BalancedIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
    #     elif sampler_name == "SetReWeightSampler":
    #         set_weight = cfg.DATALOADER.SET_WEIGHT
    #         sampler = samplers.SetReWeightSampler(train_set.img_items, mini_batch_size, num_instance, set_weight)
    #     elif sampler_name == "ImbalancedDatasetSampler":
    #         sampler = samplers.ImbalancedDatasetSampler(train_set.img_items)
    #     else:
    #         raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "train_set": train_set,
        "single_set": single_set,
        "sampler": sampler,
        "single_sampler": single_sampler,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }

def build_reid_train_loader(
        train_set, single_set, *, sampler=None, single_sampler=None, total_batch_size, num_workers=0,
):
    """
    Build a dataloader for object re-identification with some default features.
    This interface is experimental.

    Returns:
        torch.utils.data.DataLoader: a dataloader.
    """

    # mini_batch_size = total_batch_size // comm.get_world_size()

    single_batch_sampler = []
    single_train_loader = []



    # dataset = mindspore.dataset.GeneratorDataset(sampler=sampler)
    # batch_sampler = dataset.batch(mini_batch_size, True)
    
    # batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, mini_batch_size, True)
    for i in range(len(single_set)):
        # tmp_batch_sampler = mindspore.dataset.GeneratorDataset(single_sampler[i])
        # tmp_dataset = tmp_batch_sampler.batch(mini_batch_size // 2, True)
        single_batch_sampler.append(single_sampler[i])
        # single_batch_sampler.append(torch.utils.data.sampler.BatchSampler(single_sampler[i], mini_batch_size // 2, True))

    train_loader = mindspore.dataset.GeneratorDataset(
        source=train_set,
        column_names=["data"],
        # column_names=["images0", "images", "targets", "camids", "domainids", "img_paths"],
        num_parallel_workers=1,
    )
    # iterator = iter(train_loader)
    # print(next(iterator))

    # print(len(list(train_loader)))
    # exit(0)

    # 我写的
    # train_loader = DataLoaderX(
    #     comm.get_local_rank(),
    #     source=train_set,
    #     column_names=["data"],
    #     num_parallel_workers=num_workers,
    # )

    # 原来的
    # train_loader = DataLoaderX(
    #     comm.get_local_rank(),
    #     dataset=train_set,
    #     num_workers=num_workers,
    #     batch_sampler=batch_sampler,
    #     collate_fn=fast_batch_collator,
    #     pin_memory=True,
    # )
    for i in range(len(single_set)):
        single_train_loader.append(mindspore.dataset.GeneratorDataset(
            source=single_set[i],
            column_names=["data"],
            num_parallel_workers=num_workers,
        ))
        # single_train_loader.append(DataLoaderX(
        #     comm.get_local_rank(),
        #     source=single_set[i],
        #     column_names=["data"],
        #     num_parallel_workers=num_workers,
        # ))
        # single_train_loader.append(DataLoaderX(
        #     comm.get_local_rank(),
        #     dataset=single_set[i],
        #     num_workers=num_workers,
        #     batch_sampler=single_batch_sampler[i],
        #     collate_fn=fast_batch_collator,
        #     pin_memory=True,
        # ))

    return train_loader, single_train_loader

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""
def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    # model.to(torch.device(cfg.MODEL.DEVICE))
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model

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

_GradientClipperInput = Union[mindspore.Tensor, Iterable[mindspore.Tensor]]
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
        if contiguous:
            params = ContiguousParams(params)
        if solver_opt == "SGD":
            return maybe_add_freeze_layer(
                cfg,
                maybe_add_gradient_clipping(cfg, mindspore.nn.optim_ex.SGD)
                # maybe_add_gradient_clipping(cfg, mindspore.nn.optim_ex.SGD)
                # maybe_add_gradient_clipping(cfg, torch.optim.SGD)
            )(
                params,
                # params.contiguous() if contiguous else params,
                # learning_rate=0.01,
                lr=0.01,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            ), params
        else:
            return maybe_add_freeze_layer(
                cfg,
                maybe_add_gradient_clipping(cfg, getattr(torch.optim, solver_opt))
            )(params.contiguous() if contiguous else params), params

def build_lr_scheduler(cfg, optimizer, iters_per_epoch):
    max_epoch = cfg.SOLVER.MAX_EPOCH - max(
        math.ceil(cfg.SOLVER.WARMUP_ITERS / iters_per_epoch), cfg.SOLVER.DELAY_EPOCHS)

    scheduler_dict = {}

    scheduler_args = {
        "MultiStepLR": {
            "optimizer": optimizer,
            # multi-step lr scheduler options
            "milestones": cfg.SOLVER.STEPS,
            "gamma": cfg.SOLVER.GAMMA,
        },
        "CosineAnnealingLR": {
            # cosine annealing lr scheduler options
            "max_lr": cfg.SOLVER.BASE_LR,
            "min_lr": cfg.SOLVER.ETA_MIN_LR,
            "total_step": max_epoch,
            "step_per_epoch": cfg.SOLVER.STEPS[0],
            "decay_epoch": 1,
        },
        # "CosineAnnealingLR": {
        #     "optimizer": optimizer,
        #     # cosine annealing lr scheduler options
        #     "T_max": max_epoch,
        #     "eta_min": cfg.SOLVER.ETA_MIN_LR,
        # },
    }

    if cfg.SOLVER.SCHED == 'CosineAnnealingLR':
        scheduler_dict["lr_sched"] = mindspore.nn.cosine_decay_lr(**scheduler_args[cfg.SOLVER.SCHED])
    else:
        print("error")
        exit(0)
    # scheduler_dict["lr_sched"] = getattr(lr_scheduler, cfg.SOLVER.SCHED)(
    #     **scheduler_args[cfg.SOLVER.SCHED])

    if cfg.SOLVER.WARMUP_ITERS > 0:
        warmup_args = {
            "learning_rate": cfg.SOLVER.BASE_LR,
            "warmup_steps": cfg.SOLVER.WARMUP_ITERS,
        }
        scheduler_dict["warmup_sched"] = mindspore.nn.WarmUpLR(**warmup_args)
        # warmup_args = {
        #     "optimizer": optimizer,

        #     # warmup options
        #     "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
        #     "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
        #     "warmup_method": cfg.SOLVER.WARMUP_METHOD,
        # }
        # scheduler_dict["warmup_sched"] = lr_scheduler.WarmupLR(**warmup_args)

    return scheduler_dict

class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 6 methods. The way they are called is demonstrated
    in the following snippet:
    .. code-block:: python
        hook.before_train()
        for _ in range(start_epoch, max_epoch):
            hook.before_epoch()
            for iter in range(start_iter, max_iter):
                hook.before_step()
                trainer.run_step()
                hook.after_step()
            hook.after_epoch()
        hook.after_train()
    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_epoch(self):
        """
        Called before each epoch.
        """
        pass

    def after_epoch(self):
        """
        Called after each epoch.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.
    Attributes:
        iter(int): the current iteration.
        epoch(int): the current epoch.
        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.
        max_epoch (int): The epoch to end training.
        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_epoch: int, max_epoch: int, iters_per_epoch: int):
        """
        Args:
            start_epoch, max_epoch (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from epoch {}".format(start_epoch))

        self.iter = self.start_iter = start_epoch * iters_per_epoch

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.epoch in range(start_epoch, max_epoch):
                    self.before_epoch()
                    for _ in range(iters_per_epoch):
                        self.before_step()
                        self.run_step()
                        self.after_step()
                        self.iter += 1
                    self.after_epoch()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_epoch(self):
        self.storage.epoch = self.epoch

        for h in self._hooks:
            h.before_epoch()
        if self.epoch == 10:
            logger.info("Switch Epoch 10 Coming!")

    def before_step(self):
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def after_epoch(self):
        for h in self._hooks:
            h.after_epoch()

    def run_step(self):
        raise NotImplementedError

    def run_step_meta_learning1(self, epoch):
        raise NotImplementedError

    def run_step_meta_learning2(self, epoch):
        raise NotImplementedError

# def cycle(dl):
#     while True:
#         for data in dl:
#             yield data

# def dataLoader(ds, bs):
#     ds.batch(batch_size=bs)
#     return ds

class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:
    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.
    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, single_data_loader, optimizer, param_wrapper, optimizer_meta, param_wrapper_meta, iters_per_epoch):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of heads.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        # model.train()
        model.set_train(True)

        self.model = model
        self.data_loader = data_loader
        # self.data_loader = self.data_loader.batch(64, drop_remainder=True)
        self.single_data_loader = single_data_loader
        # self._data_loader_iter = dataLoader(data_loader, 32)
        # self._data_loader_iter = cycle(self._data_loader_iter)
        self._data_loader_iter = iter(data_loader)
        self._single_data_loader_iter = [iter(single_loader) for single_loader in single_data_loader]
        self.optimizer = optimizer
        self.param_wrapper = param_wrapper
        self.optimizer_meta = optimizer_meta
        self.param_wrapper_meta = param_wrapper_meta

        self.all_layers = dict() # find all parameters
        # model_dict[item.name] = item.value()
        # for name, param in self.model.named_parameters():
        for item in self.model.get_parameters():
            name = item.name
            param = item.value()
            name = '.'.join(name.split('.')[:-1])
            raw_name = copy.copy(name)
            for i in range(20):
                name = name.replace('.{}.'.format(i), '[{}].'.format(i))
            for i in range(5):
                name = name.replace('downsample.{}'.format(i), 'downsample[{}]'.format(i))
            for i in range(5):
                name = name.replace('bottleneck.{}'.format(i), 'bottleneck[{}]'.format(i))
            exist_name = False
            for name_list in self.all_layers:
                if name == name_list:
                    exist_name = True
            if not exist_name:
                self.all_layers[name] = dict()
                self.all_layers[name]['name'] = name
                self.all_layers[name]['raw_name'] = raw_name

        self.grad_name = list()
        # for name, _ in self.model.named_parameters():
        for item in self.model.get_parameters():
            name = item.name
            _ = item.value()
            for i in range(5):
                name = name.replace('.{}.'.format(i), '[{}].'.format(i))
            for i in range(2):
                name = name.replace('downsample.{}'.format(i), 'downsample[{}]'.format(i))
            for i in range(2):
                name = name.replace('bottleneck.{}'.format(i), 'bottleneck[{}]'.format(i))
            self.grad_name.append(name)

    def run_step_meta_learning1(self, epoch):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        # print('train_loop.py   run_step_meta_learning1')

        opt = self.opt_setting('basic')
        # data = self._data_loader_iter
        data = next(self._data_loader_iter)
        # data = self.data_loader.batch(32)
        # print("data.shape", len(data), data[0]['images0'].shape)
        data_time = time.perf_counter() - start
        # losses, loss_dict = self.basic_forward(data, self.model, epoch, opt) # forward

        data = data[0]
        # print("type(data['images'])", type(data["images"]))
        print("data", data)
        images0 = data["images0"]
        
        # images0 = mindspore.Tensor(data["images0"].numpy().astype(np.int32), mindspore.int32)
        images = data["images"]
        # images = mindspore.Tensor(data["images"].numpy().astype(np.int32), mindspore.int32)
        targets = data["targets"]
        # print("targets", type(targets))
        camids = data["camids"]
        # print("targets", type(targets))
        domainids = data["domainids"]
        img_paths = data["img_paths"]

        # loss_dict = model(data, epoch, opt)
        weights = self.optimizer.parameters
        grad_fn = mindspore.value_and_grad(self.model, grad_position=None, weights=weights, has_aux=False)
        opt = -1

        loss_dict, inputs_gradient = grad_fn(images0, images, targets, camids, domainids, img_paths, epoch, opt)

        # losses = sum(loss_dict.values()).mean()

        self.optimizer(inputs_gradient)

        # self.basic_backward(losses, self.optimizer)
        
        # Open this if and only if the 'run_step_meta_learnig2()' function is not exeucted
        # QUES 似乎没啥用
        # self._write_metrics(loss_dict, data_time)

        # if isinstance(self.param_wrapper, ContiguousParams):
        #     self.param_wrapper.assert_buffer_is_valid()

    def run_step_meta_learning2(self, epoch):
        start = time.perf_counter()
        metrics_dict = {}
        mtrain_losses = []
        mtest_losses = []
        source_count = len(self._single_data_loader_iter)
        metaTestID = np.random.choice(source_count)

        opt = self.opt_setting('mtrain')

        data_mtrain = {}
        for i in range(len(self._single_data_loader_iter)):
            if i == metaTestID:
                continue
            temp_data = next(self._single_data_loader_iter[i])
            for key in temp_data.keys():
                if key in data_mtrain.keys():
                    if isinstance(temp_data[key], torch.Tensor):
                        data_mtrain[key].append(temp_data[key])
                    else:
                        data_mtrain[key].extend(temp_data[key])
                else:
                    if isinstance(temp_data[key], torch.Tensor):
                        data_mtrain[key] = [temp_data[key]]
                    else:
                        data_mtrain[key] = temp_data[key]
        for key in data_mtrain.keys():
            if isinstance(temp_data[key][0], torch.Tensor):
                data_mtrain[key] = torch.cat(data_mtrain[key], 0)

        losses, loss_dict = self.basic_forward(data_mtrain, self.model, epoch, opt) # forward
        mtrain_losses.append(losses)
        for name, val in loss_dict.items():
            metrics_dict[name] = metrics_dict[name] + val if name in metrics_dict.keys() else val

        opt = self.opt_setting('mtest', losses) # auto_grad based on requires_grad of model
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            opt['grad_params'] = [param.repeat(*([num_gpus]+[1]*(len(param.shape)-1))) for param in opt['grad_params']]
        data_mtest = next(self._single_data_loader_iter[metaTestID])
        losses, loss_dict = self.basic_forward(data_mtest, self.model, epoch, opt) # forward
        mtest_losses.append(losses)
        for name, val in loss_dict.items():
            metrics_dict[name] = metrics_dict[name] + val if name in metrics_dict.keys() else val

        if len(mtrain_losses) > 0:
            mtrain_losses = mtrain_losses[0]
        if len(mtest_losses) > 0:
            mtest_losses = mtest_losses[0]

        total_losses = mtrain_losses + mtest_losses
        self.basic_backward(total_losses, self.optimizer_meta, True)
        data_time = time.perf_counter() - start
        
        self._write_metrics(metrics_dict, data_time)

        # if isinstance(self.param_wrapper_meta, ContiguousParams):
        #     self.param_wrapper_meta.assert_buffer_is_valid()

    def basic_forward(self, data, model, epoch, opt=-1):
        # print('train_loop.py   basic_forward')
        # print("targets" in data)
        # print("data", len(data), type(data), data)
        data = data[0]
        print("type(data['images'])", type(data["images"]))
        # print("data", data)

        loss_dict = model(data, epoch, opt)
        loss_dict, inputs_gradient = mindspore.value_and_grad(model)
        losses = sum(loss_dict.values()).mean()

        return losses, loss_dict

    def basic_backward(self, losses, optimizer, retain_graph=False):
        
        # torch.distributed.barrier()
        if (losses != None) and (optimizer != None):
            # print('start train_loop.py   basic_backward', torch.distributed.get_rank())
            optimizer.zero_grad()
            # print('after train_loop.py   zero_grad', torch.distributed.get_rank())
            # print(losses.device, torch.distributed.get_rank())
            losses.backward(retain_graph=retain_graph)

            optimizer.step()
            # print('train_loop.py   basic_backward', torch.distributed.get_rank())
            # torch.distributed.barrier()

    def opt_setting(self, flag, losses=None):
        if flag == 'basic':
            opt = {}
            opt['type'] = 'basic'
            opt['type_running_stats'] = 'general'
            opt['meta'] = False
        elif flag == 'mtrain':
            opt = {}
            opt['type'] = 'mtrain'
            opt['type_running_stats'] = 'hold'
            opt['meta'] = False
        elif flag == 'mtest':
            opt = {}
            opt['type'] = 'mtest'
            opt['type_running_stats'] = 'hold'
            opt['meta'] = True

            # import pdb; pdb.set_trace()
            for name, val in self.all_layers.items(): # allocate stepsize
                meta_ratio = 0.5 if 'adaptor' in name else 1
                exec('self.model.{}.{} = {}'.format(name, 'w_step_size', self.optimizer_meta.param_groups[0]['lr'] * meta_ratio))
                exec('self.model.{}.{} = {}'.format(name, 'b_step_size', self.optimizer_meta.param_groups[0]['lr'] * meta_ratio))

            names_weights_copy = dict()
            for name, param in self.model.named_parameters():
                # if 'certainty_param' in name:
                #     continue
                if param.requires_grad:
                    names_weights_copy['self.model.' + name] = param
                else:
                    if self.iter == 0:
                        logger.info("[{}] This parameter does have requires_grad".format(name))

            opt['grad_name'] = list()
            for key in names_weights_copy.keys():
                opt['grad_name'].append(key)
            self.optimizer_meta.zero_grad()
            grad_params = torch.autograd.grad(
                losses.mean(), names_weights_copy.values(),
                create_graph=True, allow_unused=True)
            grad_params = list(grad_params)
            for i in range(len(grad_params)):
                if grad_params[i] != None:
                    grad_params[i] = Variable(grad_params[i].data, requires_grad=False)
                else:
                    if self.iter == 0:
                        logger.info("[{}th grad] This parameter does have gradient".format(i))
            grad_params = tuple(grad_params)
            opt['grad_params'] = [p if p != None else None for p in grad_params ]
        
        return opt

    def run_step(self, epoch):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the heads, you can wrap the model.
        """

        loss_dict = self.model(data, epoch)
        losses = sum(loss_dict.values()).mean()

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()
        if isinstance(self.param_wrapper, ContiguousParams):
            self.param_wrapper.assert_buffer_is_valid()

    # def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float):
    #     """
    #     Args:
    #         loss_dict (dict): dict of scalar losses
    #         data_time (float): time taken by the dataloader iteration
    #     """
    #     device = next(iter(loss_dict.values())).device

    #     # Use a new stream so these ops don't wait for DDP or backward
    #     with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
    #         # metrics_dict = {k: v.mean().detach().cpu().item() for k, v in loss_dict.items()}
    #         metrics_dict = {k: v.mean().detach().cpu().item() for k, v in loss_dict.items()}
    #         metrics_dict["data_time"] = data_time

    #         # Gather metrics among all workers for logging
    #         # This assumes we do DDP-style training, which is currently the only
    #         # supported method in detectron2.
    #         all_metrics_dict = comm.gather(metrics_dict)

    #     if comm.is_main_process():
    #         storage = get_event_storage()

    #         # data_time among workers can have high variance. The actual latency
    #         # caused by data_time is the maximum among workers.
    #         data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
    #         storage.put_scalar("data_time", data_time)

    #         # average the rest metrics
    #         metrics_dict = {
    #             k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
    #         }
    #         total_losses_reduced = sum(metrics_dict.values())
    #         if not np.isfinite(total_losses_reduced):
    #             raise FloatingPointError(
    #                 f"Loss became infinite or NaN at iteration={self.iter}!\n"
    #                 f"loss_dict = {metrics_dict}"
    #             )

    #         storage.put_scalar("total_loss", total_losses_reduced)
    #         if len(metrics_dict) > 1:
    #             storage.put_scalars(**metrics_dict)


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, param_wrapper, grad_scaler=None):
        """

        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, param_wrapper)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self, epoch):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data, epoch)
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if isinstance(self.param_wrapper, ContiguousParams):
            self.param_wrapper.assert_buffer_is_valid()

def before_train(self):
    for h in self._hooks:
        h.before_train()

def after_train(self):
    self.storage.iter = self.iter
    for h in self._hooks:
        h.after_train()

def before_epoch(self):
    self.storage.epoch = self.epoch

    for h in self._hooks:
        h.before_epoch()
    if self.epoch == 10:
        logger.info("Switch Epoch 10 Coming!")

def before_step(self):
    self.storage.iter = self.iter

    for h in self._hooks:
        h.before_step()

def after_step(self):
    for h in self._hooks:
        h.after_step()

def after_epoch(self):
    for h in self._hooks:
        h.after_epoch()

def run_step(self):
    raise NotImplementedError
def main(args):
    cfg = setup(args)

    # if args.eval_only:
    #     cfg.defrost()
    #     cfg.MODEL.BACKBONE.PRETRAIN = False
    #     model = DefaultTrainer.build_model(cfg)

    #     Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    #     res = DefaultTrainer.test(cfg, model, 0)
    #     return res

    # trainer = DefaultTrainer(cfg)

    logger = logging.getLogger("fastreid")
    # if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for fastreid
    #     setup_logger()

    # Assume these objects must be constructed in this order.
    data_loader, single_data_loader = build_train_loader(cfg)
    cfg = auto_scale_hyperparams(cfg, data_loader.source.num_classes1, data_loader.source.num_classes2, data_loader.source.num_classes3)
    # cfg = self.auto_scale_hyperparams(cfg, data_loader.dataset.num_classes1, data_loader.dataset.num_classes2, data_loader.dataset.num_classes3)
    model = build_model(cfg)
    optimizer, param_wrapper = build_optimizer(cfg, model)
    optimizer_meta, param_wrapper_meta = build_optimizer(cfg, model, flag='meta') 

    # For training, wrap with DDP. But don't need this for inference.
    # if comm.get_world_size() > 1:
    #     # ref to https://github.com/pytorch/pytorch/issues/22049 to set `find_unused_parameters=True`
    #     # for part of the parameters is not updated.
    #     #CHANGE Set find_unused_parameters as True to realize the KD process 
    #     model = DistributedDataParallel(
    #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
    #     )
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU", device_id=0)
    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU", device_id=0)
    # model = torch.nn.DataParallel(model)


    iters_per_epoch = len(data_loader.source) // cfg.SOLVER.IMS_PER_BATCH
    # print("cfg.SOLVER.IMS_PER_BATCH", cfg.SOLVER.IMS_PER_BATCH)
    # data_loader = data_loader.batch(cfg.SOLVER.IMS_PER_BATCH, True, output_columns=[f"img_item{index}" for index in range(cfg.SOLVER.IMS_PER_BATCH)], per_batch_map=self.custom_per_batch_map)
    # data_loader = data_loader.batch(cfg.SOLVER.IMS_PER_BATCH, True, output_columns=["images0", "images", "targets"], per_batch_map=self.custom_per_batch_map)
    # self.iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
    #CHANGE Change here for iter-level warmup
    cfg.SOLVER.WARMUP_ITERS *= iters_per_epoch
    cfg.freeze()
    scheduler = build_lr_scheduler(cfg, optimizer, iters_per_epoch)
    # scheduler_meta = build_lr_scheduler(cfg, optimizer_meta, iters_per_epoch)

    _trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
        model, data_loader, single_data_loader, optimizer, param_wrapper, optimizer_meta, param_wrapper_meta, iters_per_epoch
    )
    # Assume no other objects need to be checkpointed.
    # We can later make it checkpoint the stateful hooks
    # checkpointer = Checkpointer(
    #     # Assume you want to save checkpoints together with logs/statistics
    #     model,
    #     cfg.OUTPUT_DIR,
    #     save_to_disk=comm.is_main_process(),
    #     optimizer=optimizer,
    #     **scheduler,
    # )

    start_epoch = 0
    max_epoch = cfg.SOLVER.MAX_EPOCH
    max_iter = max_epoch * iters_per_epoch
    warmup_iters = cfg.SOLVER.WARMUP_ITERS
    delay_epochs = cfg.SOLVER.DELAY_EPOCHS
    cfg = cfg

    # checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

    # if args.resume and checkpointer.has_checkpoint():
    #     start_epoch = checkpoint.get("epoch", -1) + 1


    logger = logging.getLogger(__name__)
    logger.info("Starting training from epoch {}".format(start_epoch))

    iter = start_iter = start_epoch * iters_per_epoch

    with start_iter as storage:
        try:
            before_train()
            for epoch in range(start_epoch, max_epoch):
                before_epoch()
                for _ in range(iters_per_epoch):
                    before_step()
                    run_step()
                    after_step()
                    iter += 1
                after_epoch()
        except Exception:
            logger.exception("Exception during training:")
            raise
        finally:
            after_train()

    # trainer.resume_or_load(resume=args.resume)
    # trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
