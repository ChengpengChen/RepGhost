# @Author  : chengpeng.chen
# @Email   : chencp@live.com
"""
RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization By Chengpeng Chen, Zichao Guo, Haien Zeng, Pengfei Xiong, and Jian Dong.
https://arxiv.org/abs/2211.06088
"""
import argparse
import os
import importlib
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from model.repghost import repghost_model_convert

parser = argparse.ArgumentParser(description='RepGhost Conversion for Inference')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-m', '--model', metavar='model', default='repghot.repghostnet_0_5x')
parser.add_argument('--ema-model', '--ema', action='store_true', help='to load the ema model')
parser.add_argument('--sanity_check', '-c', action='store_true', help='to check the outputs of the models')


def convert():
    args = parser.parse_args()

    m = importlib.import_module(f"model.{args.model.split('.')[0]}")
    train_model = getattr(m, args.model.split('.')[1])()
    train_model.eval()

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load, map_location='cpu')
        if args.ema_model and 'state_dict_ema' in checkpoint:
            checkpoint = checkpoint['state_dict_ema']
        else:
            checkpoint = checkpoint['state_dict']

        try:
            train_model.load_state_dict(checkpoint)
        except Exception as e:
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            # print(ckpt.keys())
            train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    infer_model = repghost_model_convert(train_model, save_path=args.save)
    print("=> saved checkpoint to '{}'".format(args.save))

    if args.sanity_check:
        data = torch.randn(5, 3, 224, 224)
        out = train_model(data)
        out2 = infer_model(data)
        print('=> The diff is', ((out - out2) ** 2).sum())


if __name__ == '__main__':
    convert()
