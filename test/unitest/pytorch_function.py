import torch
import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F


def update_parameter(param, step_size, opt=None, reserve=False):
    flag_update = False
    if step_size is not None:
        if param is not None:
            if opt['grad_params'][0] == None:
                if not reserve:
                    del opt['grad_params'][0]
                updated_param = param
            else:
                updated_param = param - step_size * opt['grad_params'][0]
                if not reserve:
                    del opt['grad_params'][0]
            flag_update = True
    if not flag_update:
        return param

    return updated_param

class MetaConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    
    def forward(self, inputs, opt=None):
        if opt != None and opt['meta']:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt)
            updated_bias = update_parameter(self.bias, self.b_step_size, opt)
            return F.conv2d(inputs, updated_weight, updated_bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
class MetaBNNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, bias_freeze=False, weight_init=1.0, bias_init=0.0):

        track_running_stats = True
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.bias_freeze = bias_freeze
        self.weight.requires_grad_(True)
        self.bias.requires_grad_(not bias_freeze)


    def forward(self, inputs, opt = None, reserve = False):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inputs.dim()))
        if opt != None and opt['meta']:
            use_meta_learning = True
        else:
            use_meta_learning = False

        if self.training:
            if opt != None:
                norm_type = opt['type_running_stats']
            else:
                norm_type = "hold"
        else:
            norm_type = "eval"

        if use_meta_learning and self.affine:
            updated_weight = update_parameter(self.weight, self.w_step_size, opt, reserve)
            if not self.bias_freeze:
                updated_bias = update_parameter(self.bias, self.b_step_size, opt, reserve)
            else:
                updated_bias = self.bias
        else:
            updated_weight = self.weight
            updated_bias = self.bias


        if norm_type == "general": # update, but not apply running_mean/var
            result = F.batch_norm(inputs, self.running_mean, self.running_var,
                                    updated_weight, updated_bias,
                                    self.training, self.momentum, self.eps)
        elif norm_type == "hold": # not update, not apply running_mean/var
            result = F.batch_norm(inputs, None, None,
                                    updated_weight, updated_bias,
                                    True, self.momentum, self.eps)
        elif norm_type == "eval": # fix and apply running_mean/var,
            result = F.batch_norm(inputs, self.running_mean, self.running_var,
                                    updated_weight, updated_bias,
                                    False, self.momentum, self.eps)
        return result