# import torch
# from torch import nn
# import torch.nn.functional as F

import mindspore
import mindspore.nn as nn
# from mindspore import ops
import numpy as np


def update_parameter(param, step_size, opt=-1, reserve=False):
    flag_update = False
    updated_param = None
    if step_size is not None:
        if param is not None:
            if opt is not None:
                if opt['grad_params'][0] == None:
                    if not reserve:
                        opt['grad_params'].pop(0)
                        # del opt['grad_params'][0]
                    updated_param = param
                else:
                    updated_param = param - step_size * opt['grad_params'][0]
                    if not reserve:
                        opt['grad_params'].pop(0)
                        # del opt['grad_params'][0]
            flag_update = True
    if not flag_update:
        return param

    return updated_param


# class MetaGate(nn.Module):
class MetaGate(nn.Cell):
    def __init__(self, feat_dim):
        super().__init__()
        self.gate = mindspore.Parameter(mindspore.ops.randn(feat_dim) * 0.1)
        # self.gate = nn.Parameter(torch.randn(feat_dim) * 0.1)
        self.sigmoid = nn.Sigmoid()

    # def forward(self, inputs1, inputs2, opt=-1):
    def construct(self, inputs1, inputs2, opt=-1):
        if opt != -1 and opt['meta']:
            updated_gate = self.sigmoid(update_parameter(self.gate, self.w_step_size, opt)).reshape(1, -1, 1, 1)

            return updated_gate * inputs1 + (1. - updated_gate) * inputs2
        else:
            gate = self.sigmoid(self.gate).reshape(1, -1, 1, 1)
            return gate * inputs1 + (1. - gate) * inputs2


# class MetaParam(nn.Module):
class MetaParam(nn.Cell):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.centers = mindspore.Parameter(mindspore.ops.randn(num_classes, feat_dim))
        # self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    # def forward(self, inputs, opt=-1):
    def construct(self, inputs, opt=-1):
        op = mindspore.ops.ReduceSum(keep_dims=True)
        if opt != -1 and opt['meta']:
            updated_centers = update_parameter(self.centers, self.w_step_size, opt)
            batch_size = inputs.size(0)
            num_classes = self.centers.size(0)
            distmat = op(mindspore.ops.pow(inputs, 2), axis=1).broadcast_to((batch_size, num_classes)) + \
                        op(mindspore.ops.pow(updated_centers, 2), axis=1).broadcast_to((num_classes, batch_size)).t()
            # distmat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
            #             torch.pow(updated_centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
            # distmat.addmm_(1, -2, inputs, updated_centers.t())
            distmat = mindspore.ops.addmm(distmat, inputs, self.centers.t(), beta=1, alpha=-2)

            return distmat

        else:
            batch_size = inputs.shape[0]
            # batch_size = inputs.size(0)
            num_classes = self.centers.shape[0]
            # num_classes = self.centers.size(0)
            distmat = op(mindspore.ops.pow(inputs, 2), axis=1).broadcast_to((batch_size, num_classes)) + \
                        op(mindspore.ops.pow(self.centers, 2), axis=1).broadcast_to((num_classes, batch_size)).t()
            # distmat = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
            #             torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
            # distmat.addmm_(1, -2, inputs, self.centers.t())
            distmat = mindspore.ops.addmm(distmat, inputs, self.centers.t(), beta=1, alpha=-2)

            return distmat


class MetaConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, group=1, bias=True, pad_mode='pad'):
        super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group, has_bias=bias)
        self.conv = mindspore.ops.Conv2D(out_channel=self.out_channels, kernel_size=self.kernel_size, mode=1, pad_mode=self.pad_mode, pad=self.padding, stride=self.stride, dilation=self.dilation, group=self.group)
    def construct(self, inputs, opt=-1):
        if opt != -1 and opt['meta']:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            # output = mindspore.ops.conv2d(inputs, updated_weight, updated_bias, self.stride, self.pad_mode, self.padding, self.dilation, self.group)
            output = self.conv(inputs, updated_weight)
            return output
        else:
            # output = mindspore.ops.conv2d(inputs, self.weight, self.bias, self.stride, self.pad_mode, self.padding, self.dilation, self.group)
            # conv = mindspore.ops.Conv2D(out_channel=self.out_channels, kernel_size=self.kernel_size, mode=1, pad_mode=self.pad_mode, pad=self.padding, stride=self.stride, dilation=self.dilation, group=self.group)
            output = self.conv(inputs, self.weight)
            return output

# class MetaConv2d(nn.Conv2d):
#     # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, group=1, bias=True, pad_mode='pad'):
#         super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group, has_bias=bias)
#         # super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
#         # super().__init__()
#         # super(MetaConv2d, self).__init__()
#         self.pad_mode = 'pad'
#         # self.kernel_size = kernel_size
#         self.w_step_size = 1
#         self.b_step_size = 1
#         self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, pad_mode=pad_mode, padding=padding, dilation=dilation, group=group)
#         self.conv.weight = self.weight
#         # self.conv.weight.set_data(self.weight)
#         # self.weight = mindspore.ops.randn((out_channels, in_channels, kernel_size, kernel_size))

#     # def forward(self, inputs, opt=-1):
#     def construct(self, inputs):
#         # if opt:
#         output = self.conv(inputs)
#         return output
#         # else:
#         #     updated_weight = update_parameter(self.weight, self.w_step_size, opt)
#         #     updated_bias = update_parameter(self.bias, self.b_step_size, opt)
#         #     output = self.conv(inputs)
#         # return output
#         # if opt != -1 and opt['meta']:
#         #     updated_weight = update_parameter(self.weight, self.w_step_size, opt)
#         #     updated_bias = update_parameter(self.bias, self.b_step_size, opt)
#         #     # return F.conv2d(inputs, updated_weight, updated_bias, self.stride, self.padding, self.dilation, self.groups)
#         #     # self.conv.weight = updated_weight
#         #     # self.conv.bias = updated_bias
#         #     output = self.conv(inputs)
#         #     # output = mindspore.ops.conv2d(inputs, updated_weight, updated_bias, self.stride, self.pad_mode, self.padding, self.dilation, self.group)
#         #     return output
#         # else:
#         #     # return F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         #     # self.conv.weight = self.weight
#         #     # self.conv.bias = self.bias
#         #     output = self.conv(inputs)
#         #     # output = mindspore.ops.conv2d(inputs, self.weight, self.bias, self.stride, self.pad_mode, self.padding, self.dilation, self.group)
#         #     return output


class MetaLinear(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=False):
        super().__init__(in_channels, out_channels, has_bias=has_bias)
        # super().__init__(in_feat, reduction_dim, bias=bias)
        self.linear = nn.Dense(self.in_channels, self.out_channels, has_bias=self.has_bias)
        self.linear.weight = self.weight
        # self.linear.weight.set_data(self.weight)
        # self.linear.bias.set_data(self.bias)
        # self.weight = mindspore.Parameter(mindspore.Tensor(np.random.rand(in_channels, out_channels).astype(np.float32)), name='weight')
        # self.bias = mindspore.Parameter(mindspore.Tensor(np.random.rand(out_channels).astype(np.float32)), name='bias')


    def construct(self, inputs, opt = -1, reserve = False):
        if opt != -1 and opt['meta']:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt, reserve)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt, reserve)

            # self.linear.weight.set_data(updated_weight)
            # self.linear.bias.set_data(updated_bias)
            # self.linear.weight.set_data(mindspore.common.initializer.initializer(updated_weight, self.linear.weight.shape, self.linear.weight.dtype))
            # self.linear.bias.set_data(mindspore.common.initializer.initializer(updated_bias, self.linear.bias.shape, self.linear.bias.dtype))
            # self.linear.weight = updated_weight
            # self.linear.bias = updated_bias
            output = self.linear(inputs)
            return output
            # return F.linear(inputs, updated_weight, updated_bias)
        else:
            # self.linear.weight.set_data(mindspore.common.initializer.initializer(self.weight, self.linear.weight.shape, self.linear.weight.dtype))
            # self.linear.bias.set_data(mindspore.common.initializer.initializer(self.bias, self.linear.bias.shape, self.linear.bias.dtype))
            # self.linear.weight = self.weight
            # self.linear.bias = self.bias
            output = self.linear(inputs)
            return output
            # return F.linear(inputs, self.weight, self.bias)


class MetaIBNNorm(nn.Cell):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        half1 = int(num_features / 2)
        self.half = half1
        half2 = num_features - half1
        self.IN = MetaINNorm(half1, **kwargs)
        self.BN = MetaBNNorm(half2, **kwargs)

    # def forward(self, inputs, opt=-1):
    def construct(self, inputs, opt=-1):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))

        # split = torch.split(inputs, self.half, 1)
        split = mindspore.ops.split(inputs, self.half, 1)
        # print("ms split", mindspore.ops.flatten(split[0], order='C', start_dim=0, end_dim=-1))
        # print("ms split", mindspore.ops.flatten(split[1], order='C', start_dim=0, end_dim=-1))
        out1 = self.IN(split[0], opt)
        out2 = self.BN(split[1], opt)
        # out1 = self.IN(split[0].contiguous(), opt)
        # out2 = self.BN(split[1].contiguous(), opt)
        # out = torch.cat((out1, out2), 1)
        # print(mindspore.ops.flatten(out1, order='C', start_dim=0, end_dim=-1))
        # print(mindspore.ops.flatten(out2, order='C', start_dim=0, end_dim=-1))
        out = mindspore.ops.cat((out1, out2), 1)
        # print(mindspore.ops.flatten(out, order='C', start_dim=0, end_dim=-1))
        # print()
        # out = mindspore.ops.clamp(out, min=0)
        return out


class MetaBNNorm(nn.BatchNorm2d):
    # def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, bias_freeze=False, weight_init=1.0, bias_init=0.0):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, bias_freeze=False, gamma_init='ones', beta_init='zeros'):

        # track_running_stats = True
        # super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.use_batch_statistics = True
        super().__init__(num_features, eps, momentum, affine, use_batch_statistics=self.use_batch_statistics)

        # if weight_init is not None: self.weight.data.fill_(weight_init)
        # if bias_init is not None: self.bias.data.fill_(bias_init)
        if gamma_init is not None: self.gamma = mindspore.Parameter(mindspore.common.initializer.initializer(gamma_init, self.gamma.shape, self.gamma.dtype), name="gamma", requires_grad=True)
        if beta_init is not None: self.beta = mindspore.Parameter(mindspore.common.initializer.initializer(beta_init, self.beta.shape, self.beta.dtype), name="beta", requires_grad=not bias_freeze)
        self.bias_freeze = bias_freeze
        self.affine = affine
        # self.weight.requires_grad_(True)
        # self.bias.requires_grad_(not bias_freeze)
        # self.gamma = mindspore.Parameter(mindspore.common.initializer.initializer(gamma_init, num_features), name="gamma", requires_grad=affine)


    # def forward(self, inputs, opt = -1, reserve = False):
    def construct(self, inputs, opt = -1, reserve = False):
        # print("ms self.gamma before", self.gamma.value())
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
        if opt != -1 and opt['meta']:
            use_meta_learning = True
        else:
            use_meta_learning = False

        # print("ms self.training", self.training)
        if self.training:
            if opt != -1:
                norm_type = opt['type_running_stats']
            else:
                norm_type = "hold"
        else:
            norm_type = "eval"

        if use_meta_learning and self.affine:
            updated_gamma = update_parameter(self.gamma, self.w_step_size, opt, reserve)
            if not self.bias_freeze:
                updated_beta = update_parameter(self.beta, self.b_step_size, opt, reserve)
            else:
                updated_beta = self.beta
        else:
            updated_gamma = self.gamma
            updated_beta = self.beta

        result = None
        if norm_type == "general": # update, but not apply running_mean/var
            # bn = nn.BatchNorm2d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, gamma_init=self.gamma_init, beta_init=self.beta_init, moving_mean_init=self.moving_mean, moving_var_init=self.moving_variance, use_batch_statistics=self.training)
            # bn.gamma = updated_gamma
            # bn.beta = updated_beta
            # result = bn(inputs)
            result = mindspore.ops.batch_norm(inputs, self.moving_mean, self.moving_variance,
                                updated_gamma, updated_beta,
                                self.training, self.momentum, self.eps)
        elif norm_type == "hold": # not update, not apply running_mean/var
            # bn = nn.BatchNorm2d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, gamma_init=self.gamma_init, beta_init=self.beta_init, moving_mean_init=None, moving_var_init=None, use_batch_statistics=True)
            # bn.gamma = updated_gamma
            # bn.beta = updated_beta
            # result = bn(inputs)
            result = mindspore.ops.batch_norm(inputs, None, None,
                                updated_gamma, updated_beta,
                                True, self.momentum, self.eps)
        elif norm_type == "eval": # fix and apply running_mean/var,
            # bn = nn.BatchNorm2d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.training, gamma_init=self.gamma_init, beta_init=self.beta_init, moving_mean_init=self.moving_mean, moving_var_init=self.moving_variance, use_batch_statistics=False)
            # # bn = nn.BatchNorm2d(self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, gamma_init=self.gamma_init, beta_init=self.beta_init, moving_mean_init=self.moving_mean, moving_var_init=self.moving_variance, use_batch_statistics=True)
            # bn.gamma = updated_gamma
            # bn.beta = updated_beta
            # result = bn(inputs)
            # print("ms inputs", inputs)
            # print("ms self.moving_mean", self.moving_mean)
            # print("ms updated_gamma", updated_gamma.value())
            # print("ms updated_beta", updated_beta.value())
            result = mindspore.ops.batch_norm(inputs, self.moving_mean, self.moving_variance,
                                updated_gamma, updated_beta,
                                False, self.momentum, self.eps)
            # print("ms result rwe", result)

        # if norm_type == "general": # update, but not apply running_mean/var
        #     result = F.batch_norm(inputs, self.running_mean, self.running_var,
        #                             updated_weight, updated_bias,
        #                             self.training, self.momentum, self.eps)
        # elif norm_type == "hold": # not update, not apply running_mean/var
        #     result = F.batch_norm(inputs, None, None,
        #                             updated_weight, updated_bias,
        #                             True, self.momentum, self.eps)
        # elif norm_type == "eval": # fix and apply running_mean/var,
        #     result = F.batch_norm(inputs, self.running_mean, self.running_var,
        #                             updated_weight, updated_bias,
        #                             False, self.momentum, self.eps)
        return result

# from mindspore.common.parameter import Parameter
# from mindspore.ops import operations as P
# from mindspore.common.initializer import initializer
# class MetaBNNorm(nn.Cell):
#     def __init__(self,
#                  num_features,
#                  eps=1e-5,
#                  momentum=0.9,
#                  affine=True,
#                  gamma_init='ones',
#                  beta_init='zeros',
#                  moving_mean_init='zeros',
#                  moving_var_init='ones',
#                  bias_freeze=False,
#                  use_batch_statistics=None,
#                  data_format='NCHW'):

#         super(MetaBNNorm, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.beta_init = beta_init
#         self.gamma_init = gamma_init
#         self.momentum = 1.0 - momentum
#         self.moving_mean_init = moving_mean_init
#         self.moving_var_init = moving_var_init
#         self.bn_train = P.BatchNorm(is_training=True,
#                                     epsilon=self.eps,
#                                     momentum=self.momentum,
#                                     data_format='NCHW')
#         self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps, data_format='NCHW')
#         self.moving_mean = Parameter(initializer(self.moving_mean_init, num_features), name="mean", requires_grad=False)
#         self.moving_variance = Parameter(initializer(self.moving_var_init, num_features), name="variance", requires_grad=False)
#         self.gamma = Parameter(initializer(gamma_init, num_features), name="gamma", requires_grad=affine)
#         self.beta = Parameter(initializer(beta_init, num_features), name="beta", requires_grad=affine)
#         self.bias_freeze = bias_freeze
#         self.use_batch_statistics = use_batch_statistics
#         self.affine = affine

#     def construct(self, inputs):
#         # print("inputs.ndim", inputs.ndim)
#         if inputs.ndim != 4:
#         # if inputs.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'.format(inputs.ndim))
#         # inputs = mindspore.Tensor(inputs)
#         use_meta_learning = False
#         # if opt != -1 and opt['meta']:
#         #     use_meta_learning = True
#         # else:
#         #     use_meta_learning = False

#         if self.training:
#             norm_type = "hold"
#             # if opt != -1:
#             #     norm_type = opt['type_running_stats']
#             # else:
#             #     norm_type = "hold"
#         else:
#             norm_type = "eval"

#         updated_gamma = self.gamma
#         updated_beta = self.beta
#         # if use_meta_learning and self.affine:
#         #     updated_gamma = update_parameter(self.gamma, self.w_step_size, opt, reserve)
#         #     if not self.bias_freeze:
#         #         updated_beta = update_parameter(self.beta, self.b_step_size, opt, reserve)
#         #     else:
#         #         updated_beta = self.beta
#         # else:
#         #     updated_gamma = self.gamma
#         #     updated_beta = self.beta

#         if norm_type == "general": # update, but not apply running_mean/var
#             if self.use_batch_statistics is None:
#                 if self.training:
#                     return self.bn_train(inputs, updated_gamma, updated_beta, self.moving_mean, self.moving_variance)[0]
#                 if not self.training:
#                     return self.bn_infer(inputs,
#                                      self.gamma,
#                                      self.beta,
#                                      self.moving_mean,
#                                      self.moving_variance)[0]
#             if self.use_batch_statistics:
#                 return self.bn_train(inputs, updated_gamma, updated_beta, self.moving_mean, self.moving_variance)[0]
#             return self.bn_infer(inputs,
#                                 self.gamma,
#                                 self.beta,
#                                 self.moving_mean,
#                                 self.moving_variance)[0]
#         elif norm_type == "hold": # not update, not apply running_mean/var
#             if self.use_batch_statistics is None:
#                 if self.training:
#                     # print("inputs", type(inputs), inputs)
#                     # print("updated_gamma", type(updated_gamma), updated_gamma)
#                     # print("updated_beta", type(updated_beta), updated_beta)
#                     # print("num_features", self.num_features)
#                     # print("affine", self.affine)
#                     mean = mindspore.Tensor(np.zeros([self.num_features]), mindspore.float32)
#                     variance = mindspore.Tensor(np.zeros([self.num_features]), mindspore.float32)
#                     result = self.bn_train(inputs, updated_gamma, updated_beta, mean, variance)[0]
#                     # result = self.bn_train(inputs, updated_gamma, updated_beta, None, None)[0]
#                     # result = mindspore.Tensor(result)
#                     # print("self.bn_train(inputs, updated_gamma, updated_beta, None, None)[0]", type(result), result.shape, result)
#                     return result
#                     # return self.bn_train(inputs, updated_gamma, updated_beta, None, None)[0]
#                 if not self.training:
#                     return self.bn_infer(inputs,
#                                      self.gamma,
#                                      self.beta, None, None)[0]
#             if self.use_batch_statistics:
#                     return self.bn_train(inputs, updated_gamma, updated_beta, None, None)[0]
#             return self.bn_infer(inputs,
#                                 self.gamma,
#                                 self.beta, None, None)[0]
#         elif norm_type == "eval": # fix and apply running_mean/var,
#             if self.use_batch_statistics is None:
#                 if self.training:
#                     return self.bn_train(inputs, updated_gamma, updated_beta, self.moving_mean, self.moving_variance)[0]
#                 if not self.training:
#                     return self.bn_infer(inputs,
#                                      self.gamma,
#                                      self.beta,
#                                      self.moving_mean,
#                                      self.moving_variance)[0]
#             if self.use_batch_statistics:
#                 return self.bn_train(inputs, updated_gamma, updated_beta, self.moving_mean, self.moving_variance)[0]
#             return self.bn_infer(inputs,
#                                 self.gamma,
#                                 self.beta,
#                                 self.moving_mean,
#                                 self.moving_variance)[0]
#         # return result


# from mindspore import _checkparam as validator
# class MetaINNorm(nn.Cell):
#     def __init__(self,
#                  num_features,
#                  eps=1e-5,
#                  momentum=0.1,
#                  affine=True,
#                  gamma_init='ones',
#                  beta_init='zeros'):
#         """Initialize Normalization base class."""
#         super(MetaINNorm, self).__init__()
#         validator.check_value_type('num_features', num_features, [int], self.cls_name)
#         validator.check_value_type('eps', eps, [float], self.cls_name)
#         validator.check_value_type('momentum', momentum, [float], self.cls_name)
#         validator.check_value_type('affine', affine, [bool], self.cls_name)
#         # args_input = {"gamma_init": gamma_init, "beta_init": beta_init}
#         # self.check_types_valid(args_input, 'InstanceNorm2d')
#         if num_features < 1:
#             raise ValueError(f"For '{self.cls_name}', the 'num_features' must be at least 1, but got {num_features}.")

#         if momentum < 0 or momentum > 1:
#             raise ValueError(f"For '{self.cls_name}', the 'momentum' must be a number in range [0, 1], "
#                              f"but got {momentum}.")
#         self.num_features = num_features
#         self.eps = eps
#         self.moving_mean = Parameter(initializer('zeros', num_features), name="mean", requires_grad=False)
#         self.moving_variance = Parameter(initializer('ones', num_features), name="variance", requires_grad=False)
#         self.gamma = Parameter(initializer(
#             gamma_init, num_features), name="gamma", requires_grad=affine)
#         self.beta = Parameter(initializer(
#             beta_init, num_features), name="beta", requires_grad=affine)

#         self.shape = P.Shape()
#         self.momentum = momentum
#         self.instance_bn = P.InstanceNorm(epsilon=self.eps, momentum=self.momentum)
#         self.in_fc_multiply = 0.0

#     def construct(self, inputs, opt=-1):
#         if inputs.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))

#         if (inputs.shape[2] == 1) and (inputs.shape[2] == 1):
#             inputs[:] *= self.in_fc_multiply
#             return inputs
#         else:
#             if opt != -1 and opt['meta']:
#                 use_meta_learning = True
#             else:
#                 use_meta_learning = False

#             if use_meta_learning and self.affine:
#                 updated_gamma = update_parameter(self.gamma, self.w_step_size, opt)
#                 updated_beta = update_parameter(self.beta, self.b_step_size, opt)
#             else:
#                 updated_gamma = self.gamma
#                 updated_beta = self.beta

#             result = self.instance_bn(inputs, updated_gamma, updated_beta, self.moving_mean, self.moving_variance)[0]
#             return result

class MetaINNorm(nn.InstanceNorm2d):
    # def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, bias_freeze=False, weight_init=1.0, bias_init=0.0):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, bias_freeze=False, gamma_init="ones", beta_init="zeros"):

        # track_running_stats = False
        # use_batch_statistics = False
        # super().__init__(num_features, eps, momentum, affine, track_running_stats)
        super().__init__(num_features, eps, momentum, affine)
        self.affine = affine

        if self.gamma is not None:
            if gamma_init is not None: self.gamma = mindspore.Parameter(mindspore.common.initializer.initializer(gamma_init, self.gamma.shape, self.gamma.dtype), name="gamma", requires_grad=True)
        if self.beta is not None:
            if beta_init is not None: self.beta = mindspore.Parameter(mindspore.common.initializer.initializer(beta_init, self.beta.shape, self.beta.dtype), name="beta", requires_grad=not bias_freeze)
        # if self.weight is not None:
        #     if weight_init is not None: self.weight.data.fill_(weight_init)
        #     self.weight.requires_grad_(True)
        # if self.bias is not None:
        #     if bias_init is not None: self.bias.data.fill_(bias_init)
        #     self.bias.requires_grad_(not bias_freeze)
        self.in_fc_multiply = 0.0

    # def forward(self, inputs, opt=-1):
    def construct(self, inputs, opt=-1):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))

        if (inputs.shape[2] == 1) and (inputs.shape[2] == 1):
            inputs[:] *= self.in_fc_multiply
            return inputs
        else:
            if opt != -1 and opt['meta']:
                use_meta_learning = True
            else:
                use_meta_learning = False

            if use_meta_learning and self.affine:
                updated_gamma = update_parameter(self.gamma, self.w_step_size, opt)
                updated_beta = update_parameter(self.beta, self.b_step_size, opt)
            else:
                updated_gamma = self.gamma
                updated_beta = self.beta

            # self.moving_mean = None
            # if self.moving_mean is None:
            # net = nn.InstanceNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine)
            net = nn.InstanceNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum, affine=True, gamma_init=updated_gamma, beta_init=updated_beta)
            return net(inputs)
            # if self.moving_mean is None:
                # return F.instance_norm(inputs, None, None,
                #                         updated_weight, updated_bias,
                #                         True, self.momentum, self.eps)