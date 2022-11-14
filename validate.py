# @Author  : chengpeng.chen
# @Email   : chencp@live.com
"""
RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization By Chengpeng Chen, Zichao Guo, Haien Zeng, Pengfei Xiong, and Jian Dong.
https://arxiv.org/abs/2211.06088
"""
#!/usr/bin/env python3
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
import importlib

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import safe_model_name, load_checkpoint, \
    convert_splitbn_model
from timm.utils import *


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument(
    '-c',
    '--config',
    default='',
    type=str,
    metavar='FILE',
    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')

# Dataset / Model parameters
parser.add_argument(
    '--data_dir',
    metavar='DIR',
    default='/disk2/datasets/imagenet',
    help='path to dataset')
parser.add_argument(
    '--dataset',
    '-d',
    metavar='NAME',
    default='',
    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument(
    '--train-split',
    metavar='NAME',
    default='train',
    help='dataset train split (default: train)')
parser.add_argument(
    '--val-split',
    metavar='NAME',
    default='val',
    help='dataset validation split (default: val)')
parser.add_argument(
    '--model',
    default='',
    type=str,
    metavar='MODEL',
    help='Name of model to train (default: "countception"')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='Resume full model and optimizer state from checkpoint (default: none)'
)
parser.add_argument(
    '--num-classes',
    type=int,
    default=1000,
    metavar='N',
    help='number of label classes (Model default if None)')
parser.add_argument(
    '--gp',
    default=None,
    type=str,
    metavar='POOL',
    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.'
)
parser.add_argument(
    '--img-size',
    type=int,
    default=None,
    metavar='N',
    help='Image patch size (default: None => model default)')
parser.add_argument(
    '--input-size',
    default=None,
    nargs=3,
    type=int,
    metavar='N N N',
    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty'
)
parser.add_argument(
    '--crop-pct',
    default=None,
    type=float,
    metavar='N',
    help='Input image center crop percent (for validation only)')
parser.add_argument(
    '--mean',
    type=float,
    nargs='+',
    default=None,
    metavar='MEAN',
    help='Override mean pixel value of dataset')
parser.add_argument(
    '--std',
    type=float,
    nargs='+',
    default=None,
    metavar='STD',
    help='Override std deviation of of dataset')
parser.add_argument(
    '--interpolation',
    default='',
    type=str,
    metavar='NAME',
    help='Image resize interpolation type (overrides model)')
parser.add_argument(
    '-b',
    '--batch-size',
    type=int,
    default=32,
    metavar='N',
    help='input batch size for training (default: 32)')
parser.add_argument(
    '-vb',
    '--validation-batch-size-multiplier',
    type=int,
    default=1,
    metavar='N',
    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument(
    '--aug-splits',
    type=int,
    default=0,
    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument(
    '--drop',
    type=float,
    default=0.2,
    metavar='PCT',
    help='Dropout rate (default: 0.)')
parser.add_argument(
    '--drop-connect',
    type=float,
    default=None,
    metavar='PCT',
    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument(
    '--drop-path',
    type=float,
    default=None,
    metavar='PCT',
    help='Drop path rate (default: None)')
parser.add_argument(
    '--drop-block',
    type=float,
    default=None,
    metavar='PCT',
    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument(
    '--bn-tf',
    action='store_true',
    default=False,
    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)'
)
parser.add_argument(
    '--bn-momentum',
    type=float,
    default=None,
    help='BatchNorm momentum override (if not None)')
parser.add_argument(
    '--bn-eps',
    type=float,
    default=None,
    help='BatchNorm epsilon override (if not None)')
parser.add_argument(
    '--sync-bn',
    action='store_true',
    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument(
    '--dist-bn',
    type=str,
    default='',
    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")'
)
parser.add_argument(
    '--split-bn',
    action='store_true',
    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument(
    '--model-ema',
    action='store_true',
    default=False,
    help='Enable tracking moving average of model weights')
parser.add_argument(
    '--model-ema-force-cpu',
    action='store_true',
    default=False,
    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.'
)
parser.add_argument(
    '--model-ema-decay',
    type=float,
    default=0.9998,
    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument(
    '--seed',
    type=int,
    default=42,
    metavar='S',
    help='random seed (default: 42)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=50,
    metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--recovery-interval',
    type=int,
    default=0,
    metavar='N',
    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument(
    '--checkpoint-hist',
    type=int,
    default=10,
    metavar='N',
    help='number of checkpoints to keep (default: 10)')
parser.add_argument(
    '-j',
    '--workers',
    type=int,
    default=4,
    metavar='N',
    help='how many training processes to use (default: 1)')
parser.add_argument(
    '--save-images',
    action='store_true',
    default=False,
    help='save images of input bathes every log interval for debugging')
parser.add_argument(
    '--amp',
    action='store_true',
    default=False,
    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument(
    '--apex-amp',
    action='store_true',
    default=False,
    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument(
    '--native-amp',
    action='store_true',
    default=False,
    help='Use Native Torch AMP mixed precision')
parser.add_argument(
    '--channels-last',
    action='store_true',
    default=False,
    help='Use channels_last memory layout')
parser.add_argument(
    '--pin-mem',
    action='store_true',
    default=False,
    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
)
parser.add_argument(
    '--no-prefetcher',
    action='store_true',
    default=False,
    help='disable fast prefetcher')
parser.add_argument(
    '--output',
    default='./',
    type=str,
    metavar='PATH',
    help='path to output folder (default: none, current dir)')
parser.add_argument(
    '--experiment',
    default='',
    type=str,
    metavar='NAME',
    help='name of train experiment, name of sub-folder for output')
parser.add_argument(
    '--eval-metric',
    default='top1',
    type=str,
    metavar='EVAL_METRIC',
    help='Best metric (default: "top1"')
parser.add_argument(
    '--tta',
    type=int,
    default=0,
    metavar='N',
    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)'
)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    '--use-multi-epochs-loader',
    action='store_true',
    default=False,
    help='use the multi-epochs-loader to save time at the beginning of every epoch'
)
parser.add_argument(
    '--torchscript',
    dest='torchscript',
    action='store_true',
    help='convert model torchscript for inference')
parser.add_argument(
    '--log-wandb',
    action='store_true',
    default=False,
    help='log training and validation metrics to wandb')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            'Testing in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
            % (args.rank, args.world_size))
    else:
        _logger.info('Testing with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    m = importlib.import_module(f"model.{args.model.split('.')[0]}")
    model = getattr(m, args.model.split('.')[1])(dropout=args.drop)

    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'
        ), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}'
        )

    data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.'
            )

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    amp_autocast = suppress  # do nothing

    load_checkpoint(model, args.resume, use_ema=False)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)
        load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(
                model, device_ids=[args.local_rank
                                   ])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        batch_size=args.batch_size)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem, )

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    eval_metrics = validate(
        model,
        loader_eval,
        validate_loss_fn,
        args,
        amp_autocast=amp_autocast)
    if model_ema is not None and not args.model_ema_force_cpu:
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'
                                                 ):
            distribute_bn(model_ema, args.world_size,
                          args.dist_bn == 'reduce')
        ema_eval_metrics = validate(
            model_ema.module,
            loader_eval,
            validate_loss_fn,
            args,
            amp_autocast=amp_autocast,
            log_suffix=' (EMA)')
        eval_metrics = ema_eval_metrics

    if args.rank == 0:
        print('eval_metrics = {}'.format(eval_metrics))
        time.sleep(3)


def validate(model,
             loader,
             loss_fn,
             args,
             amp_autocast=suppress,
             log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(
                    dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or
                                         batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name,
                        batch_idx,
                        last_idx,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                        top5=top5_m))

    metrics = OrderedDict(
        [('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
