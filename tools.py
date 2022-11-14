# @Author  : chengpeng.chen
# @Email   : chencp@live.com
"""
RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization By Chengpeng Chen, Zichao Guo, Haien Zeng, Pengfei Xiong, and Jian Dong.
https://arxiv.org/abs/2211.06088
"""
import torch

from infotool.profile import profile_origin
from infotool.helper import clever_format

import copy

def convert_syncbn_to_bn(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_syncbn_to_bn(child)
        )
    del module
    return module_output


def cal_flops_params(original_model, input_size):
    model = copy.deepcopy(original_model)
    model = convert_syncbn_to_bn(model)
    input_size = list(input_size)
    assert len(input_size) in [3, 4]
    if len(input_size) == 4:
        if input_size[0] != 1:
            print('modify batchsize of input_size from {} to 1'.format(input_size[0]))
            input_size[0] = 1

    if len(input_size) == 3:
        input_size.insert(0, 1)

    flops, params = profile_origin(model, inputs=(torch.zeros(input_size), ))

    print('flops = {}, params = {}'.format(flops, params))
    print('flops = {}, params = {}'.format(clever_format(flops), clever_format(params)))

    return flops, params
