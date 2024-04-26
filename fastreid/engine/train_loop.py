# encoding: utf-8
"""
credit:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
"""

import logging
import time
import weakref
from typing import Dict

import copy
import numpy as np
# import torch
# from torch.autograd import Variable
# from torch.nn.parallel import DataParallel, DistributedDataParallel
import mindspore
from mindspore.communication import get_group_size

import fastreid.utils.comm as comm
from fastreid.utils.events import EventStorage, get_event_storage
from fastreid.utils.params import ContiguousParams

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]

logger = logging.getLogger(__name__)


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

def get_gpu_num():
    import argparse
    parser = argparse.ArgumentParser(description="处理命令行参数")
    parser.add_argument('--num-gpus', type=int, default=1, help='GPU的数量')
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    args = parser.parse_args()
    return args.num_gpus

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
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)
        # data = self.data_loader.batch(32)
        # print("data.shape", len(data), data[0]['images0'].shape)
        data_time = time.perf_counter() - start
        # losses, loss_dict = self.basic_forward(data, self.model, epoch, opt) # forward

        # data = data[0]
        # print("type(data['images'])", type(data["images"]))
        # print("data", len(data), data)
        images0 = data[0]
        images = data[1]
        targets = data[2]
        # print("targets", type(targets), targets)
        camids = data[3]
        domainids = data[4]
        img_paths = data[5]
        # images0 = data["images0"]
        # images = data["images"]
        # targets = data["targets"]
        # camids = data["camids"]
        # domainids = data["domainids"]
        # img_paths = data["img_paths"]

        # loss_dict = model(data, epoch, opt)
        weights = self.optimizer.parameters
        # print("weights", type(weights), weights)

        grad_fn = mindspore.value_and_grad(self.model, grad_position=None, weights=weights, has_aux=False)

        loss_dict, inputs_gradient = grad_fn(images, targets, domainids, epoch, opt)
        # loss_dict, inputs_gradient = grad_fn(images0, images, targets, camids, domainids, img_paths, epoch, opt)

        # losses = sum(loss_dict.values()).mean()

        # inputs_gradient = list(inputs_gradient)
        # for index, i in enumerate(inputs_gradient):
        #     if i.dtype == mindspore.int64:
        #         inputs_gradient[index] = mindspore.Tensor(inputs_gradient[index], dtype=mindspore.float32)
        # inputs_gradient = tuple(inputs_gradient)
        # print("inputs_gradient", type(inputs_gradient), inputs_gradient)

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
        keys = ["images0", "images", "targets", "camids", "domainids", "img_paths"]
        for i in range(len(self._single_data_loader_iter)):
            if i == metaTestID:
                continue
            temp_data = None
            try:
                temp_data = next(self._single_data_loader_iter[i])
            except StopIteration:
                self._single_data_loader_iter = [iter(single_loader) for single_loader in self.single_data_loader]
                temp_data = next(self._single_data_loader_iter[i])
            for index, key in enumerate(keys):
                if key == "img_paths":
                    continue
                if key in data_mtrain.keys():
                    if isinstance(temp_data[index], mindspore.Tensor):
                        data_mtrain[key].append(temp_data[index])
                    else:
                        data_mtrain[key].extend(temp_data[index])
                else:
                    if isinstance(temp_data[index], mindspore.Tensor):
                        data_mtrain[key] = [temp_data[index]]
                    else:
                        data_mtrain[key] = temp_data[index]
        for key in data_mtrain.keys():
            if isinstance(temp_data[index][0], mindspore.Tensor):
                data_mtrain[key] = mindspore.ops.cat(data_mtrain[key], 0)

        # print("data_mtrain", data_mtrain["images0"].shape, data_mtrain["targets"].shape, data_mtrain["domainids"].shape, data_mtrain)


        # for i in range(len(self._single_data_loader_iter)):
        #     if i == metaTestID:
        #         continue
        #     temp_data = next(self._single_data_loader_iter[i])
        #     # print("temp_data", type(temp_data), temp_data)
        #     for index, item in enumerate(temp_data):
        #         print("temp_data[index]", temp_data[index].shape)
        #         if index < len(data_mtrain):
        #             if isinstance(temp_data[index], mindspore.Tensor):
        #                 data_mtrain[index].append(temp_data[index])
        #             else:
        #                 data_mtrain[index].extend(temp_data[index])
        #         else:
        #             if isinstance(temp_data[index], mindspore.Tensor):
        #                 data_mtrain.append([temp_data[index]])
        #             else:
        #                 data_mtrain.append(temp_data[index])
        # for index, item in enumerate(data_mtrain):
        #     if isinstance(data_mtrain[index][0], mindspore.Tensor) and data_mtrain[index][0].dtype != mindspore.string:
        #         # print("data_mtrain[index]", type(data_mtrain[index]), len(data_mtrain[index]), type(data_mtrain[index][0]), data_mtrain[index][0].shape)
        #         data_mtrain[index] = mindspore.ops.stack(data_mtrain[index], 0)
        #         # print("data_mtrain[index] 2", type(data_mtrain[index]), data_mtrain[index].shape, type(data_mtrain[index][0]), data_mtrain[index][0].shape)


        # data_mtrain = {}
        # for i in range(len(self._single_data_loader_iter)):
        #     if i == metaTestID:
        #         continue
        #     temp_data = next(self._single_data_loader_iter[i])
        #     print("temp_data", type(temp_data), temp_data)
        #     for key in temp_data.keys():
        #         if key in data_mtrain.keys():
        #             if isinstance(temp_data[key], torch.Tensor):
        #                 data_mtrain[key].append(temp_data[key])
        #             else:
        #                 data_mtrain[key].extend(temp_data[key])
        #         else:
        #             if isinstance(temp_data[key], torch.Tensor):
        #                 data_mtrain[key] = [temp_data[key]]
        #             else:
        #                 data_mtrain[key] = temp_data[key]
        # for key in data_mtrain.keys():
        #     if isinstance(temp_data[key][0], torch.Tensor):
        #         data_mtrain[key] = torch.cat(data_mtrain[key], 0)

        losses, loss_dict, inputs_gradient = self.basic_forward(data_mtrain, self.model, epoch, opt) # forward
        mtrain_losses.append(losses)
        for name, val in loss_dict.items():
            metrics_dict[name] = metrics_dict[name] + val if name in metrics_dict.keys() else val

        opt = self.opt_setting('mtest', losses) # auto_grad based on requires_grad of model
        # num_gpus = torch.cuda.device_count()
        num_gpus = get_gpu_num()
        # num_gpus = get_group_size()
        # print("num_gpus", num_gpus)
        if num_gpus > 1:
            opt['grad_params'] = [param.tile(tuple([num_gpus]+[1]*(len(param.shape)-1))) for param in opt['grad_params']]
            # opt['grad_params'] = [param.repeat(*([num_gpus]+[1]*(len(param.shape)-1))) for param in opt['grad_params']]
        data_mtest = next(self._single_data_loader_iter[metaTestID])
        data_mtest = {key: value for key, value in zip(keys, data_mtest)}
        losses, loss_dict, inputs_gradient = self.basic_forward(data_mtest, self.model, epoch, opt) # forward
        mtest_losses.append(losses)
        for name, val in loss_dict.items():
            metrics_dict[name] = metrics_dict[name] + val if name in metrics_dict.keys() else val

        if len(mtrain_losses) > 0:
            mtrain_losses = mtrain_losses[0]
        if len(mtest_losses) > 0:
            mtest_losses = mtest_losses[0]

        total_losses = mtrain_losses + mtest_losses
        self.basic_backward(inputs_gradient, self.optimizer_meta, True)
        # self.basic_backward(total_losses, self.optimizer_meta, True)
        data_time = time.perf_counter() - start

        # self._write_metrics(metrics_dict, data_time)

        # if isinstance(self.param_wrapper_meta, ContiguousParams):
        #     self.param_wrapper_meta.assert_buffer_is_valid()

    def basic_forward(self, data, model, epoch, opt=None):
        # print('train_loop.py   basic_forward')
        # print("targets" in data)
        # print("data", len(data), type(data), data)
        # data = data[0]
        # print("type(data['images'])", type(data["images"]))
        # print("data", data)
        images0 = data["images0"]
        images = data["images"]
        # print("images", images.shape)
        targets = data["targets"]
        # print("targets", targets.shape)
        camids = data["camids"]
        domainids = data["domainids"]
        # print("domainids", domainids.shape)
        # img_paths = data["img_paths"]

        weights = self.optimizer.trainable_params()
        # weights = self.optimizer.parameters
        # print("self.optimizer", self.optimizer)
        # print("self.optimizer.parameters", self.optimizer.trainable_params())
        grad_fn = mindspore.value_and_grad(model, grad_position=None, weights=weights, has_aux=False)
        (loss_cls, loss_center, loss_triplet, loss_circle, loss_cosface, loss_triplet_add, loss_triplet_mtrain, loss_stc, loss_triplet_mtest, loss_domain_intra, loss_domain_inter, loss_Center), inputs_gradient = grad_fn(images, targets, domainids, epoch, opt)

        loss_dict = {}
        # 使用locals()函数获取当前局部变量的字典
        for var_name in ['loss_cls', 'loss_center', 'loss_triplet', 'loss_circle', 'loss_cosface', 'loss_triplet_add', 'loss_triplet_mtrain', 'loss_stc', 'loss_triplet_mtest', "loss_domain_intra", "loss_domain_inter", "loss_Center"]:
            value = locals()[var_name]
            if value != 0:
                loss_dict[var_name] = value

        # loss_dict, inputs_gradient = grad_fn(images, targets, domainids, epoch, opt)
        # loss_dict = model(data, epoch, opt)
        losses = sum(loss_dict.values()).mean()
        # losses = (loss_cls + loss_center + loss_triplet + loss_circle + loss_cosface + loss_triplet_add + loss_triplet_mtrain + loss_stc + loss_triplet_mtest + loss_domain_intra + loss_domain_inter + loss_Center) / 12

        return losses, loss_dict, inputs_gradient

    def basic_backward(self, inputs_gradient, optimizer, retain_graph=False):
    # def basic_backward(self, losses, optimizer, retain_graph=False):

        if (inputs_gradient != None) and (optimizer != None):
            optimizer(inputs_gradient)
        # torch.distributed.barrier()
        # if (losses != None) and (optimizer != None):
        #     optimizer.zero_grad()
        #     losses.backward(retain_graph=retain_graph)
        #     optimizer.step()
        #     # torch.distributed.barrier()

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
            for item in self.model.trainable_params():
                names_weights_copy['self.model.' + item.name] = item.value()
            # for name, param in self.model.named_parameters():
            #     # if 'certainty_param' in name:
            #     #     continue
            #     if param.requires_grad:
            #         names_weights_copy['self.model.' + name] = param
            #     else:
            #         if self.iter == 0:
            #             logger.info("[{}] This parameter does have requires_grad".format(name))

            opt['grad_name'] = list()
            for key in names_weights_copy.keys():
                opt['grad_name'].append(key)

            grad_params = mindspore.grad(lambda x: x, grad_position=None, weights=self.model.trainable_params(), has_aux=False)(losses.mean())
            grad_params = list(grad_params)
            for i in range(len(grad_params)):
                if grad_params[i] != None:
                    grad_params[i] = mindspore.Parameter(grad_params[i].value(), requires_grad=False)
                else:
                    if self.iter == 0:
                        logger.info("[{}th grad] This parameter does have gradient".format(i))
            grad_params = tuple(grad_params)
            opt['grad_params'] = [p if p != None else None for p in grad_params ]

            # self.optimizer_meta.zero_grad()
            # grad_params = torch.autograd.grad(
            #     losses.mean(), names_weights_copy.values(),
            #     create_graph=True, allow_unused=True)
            # grad_params = list(grad_params)
            # for i in range(len(grad_params)):
            #     if grad_params[i] != None:
            #         grad_params[i] = Variable(grad_params[i].data, requires_grad=False)
            #     else:
            #         if self.iter == 0:
            #             logger.info("[{}th grad] This parameter does have gradient".format(i))
            # grad_params = tuple(grad_params)
            # opt['grad_params'] = [p if p != None else None for p in grad_params ]

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

        # self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()
        if isinstance(self.param_wrapper, ContiguousParams):
            self.param_wrapper.assert_buffer_is_valid()

    # # def _write_metrics(self, loss_dict: Dict[str, mindspore.Tensor], data_time: float):
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

        # self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if isinstance(self.param_wrapper, ContiguousParams):
            self.param_wrapper.assert_buffer_is_valid()
