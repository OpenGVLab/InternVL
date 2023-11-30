# --------------------------------------------------------
# InternVL
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import datetime
import os
import random
import subprocess
import time

import deepspeed
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from config import get_config
from dataset import build_loader
from ddp_hooks import fp16_compress_hook
from ema_deepspeed import EMADeepspeed
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import set_weight_decay_and_lr
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy
from utils import MyAverageMeter, load_pretrained, reduce_tensor


def parse_option():
    parser = argparse.ArgumentParser(
        'InternVL training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar='FILE', help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')

    # easy config modification
    parser.add_argument('--batch-size', type=int, help='batch size for single GPU')
    parser.add_argument('--dataset', type=str, help='dataset name', default=None)
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
                        )
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='work_dirs', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)'
                        )

    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--save-ckpt-num', default=1, type=int)
    parser.add_argument('--accumulation-steps', type=int, default=1, help='gradient accumulation steps')

    # distributed training
    parser.add_argument('--local-rank', type=int, required=True, help='local rank for DistributedDataParallel')

    # deepspeed config
    parser.add_argument('--disable-grad-scalar', action='store_true', help='disable Grad Scalar')
    parser.add_argument('--offload-optimizer', type=str, default='none', choices=['cpu', 'none'],
                        help='enable optimizer offloading')
    parser.add_argument('--offload-param', type=str, default='none', choices=['cpu', 'none'],
                        help='enable model offloading')
    # To use Zero3, Please use main_accelerate.py instead.
    # For this script, we are facing a similar issue as https://github.com/microsoft/DeepSpeed/issues/3068
    parser.add_argument('--zero-stage', type=int, default=1, choices=[1, 2], help='deep speed zero stage')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def seed_everything(seed, rank):
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def save_config(config):
    path = os.path.join(config.OUTPUT, 'config.json')
    with open(path, 'w') as f:
        f.write(config.dump())
    logger.info(f'Full config saved to {path}')


def build_criterion(config):
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


def scale_learning_rate(config, num_processes):
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * num_processes / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * num_processes / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * num_processes / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    logger.info('BASE_LR={}'.format(config.TRAIN.BASE_LR))
    logger.info('WARMUP_LR={}'.format(config.TRAIN.WARMUP_LR))
    logger.info('MIN_LR={}'.format(config.TRAIN.MIN_LR))


def log_model_statistic(model_wo_ddp):
    n_parameters = sum(p.numel() for p in model_wo_ddp.parameters()
                       if p.requires_grad)
    logger.info(f'number of params: {n_parameters / 1e6} M')
    if hasattr(model_wo_ddp, 'flops'):
        flops = model_wo_ddp.flops()
        logger.info(f'number of GFLOPs: {flops / 1e9}')


def get_parameter_groups(model, config):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = set_weight_decay_and_lr(
        model,
        config.TRAIN.WEIGHT_DECAY,
        config.TRAIN.BASE_LR,
        skip,
        skip_keywords,
        lr_layer_decay=config.TRAIN.LR_LAYER_DECAY,
        lr_layer_decay_ratio=config.TRAIN.LR_LAYER_DECAY_RATIO,
        freeze_backbone=config.TRAIN.OPTIMIZER.FREEZE_BACKBONE,
        dcn_lr_mul=config.TRAIN.OPTIMIZER.DCN_LR_MUL,
    )
    return parameters


def get_optimizer_state_str(optimizer):
    states = []
    for param_group in optimizer.param_groups:
        states.append(f'name={param_group["name"]} lr={param_group["lr"]} weight_decay={param_group["weight_decay"]}')
    return '\n'.join(states)


def build_ds_config(config, args):
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    if opt_lower == 'adamw':
        optimizer = {
            'type': 'AdamW',
            'params': {
                'lr': config.TRAIN.BASE_LR,
                'eps': config.TRAIN.OPTIMIZER.EPS,
                'betas': config.TRAIN.OPTIMIZER.BETAS,
                'weight_decay': config.TRAIN.WEIGHT_DECAY
            }
        }
    else:
        return NotImplemented

    ds_config = {
        'train_micro_batch_size_per_gpu': config.DATA.BATCH_SIZE,
        'optimizer': optimizer,
        'bf16': {
            'enabled': True,
        },
        'zero_optimization': {
            'stage': 1,
            'allgather_partitions': True,
            'allgather_bucket_size': 1e9,
            'overlap_comm': True,
            'reduce_scatter': True,
            'reduce_bucket_size': 1e9,
            'contiguous_gradients': True
        },
        'steps_per_print': 1e10,
        'gradient_accumulation_steps': config.TRAIN.ACCUMULATION_STEPS,
        'gradient_clipping': config.TRAIN.CLIP_GRAD,
    }
    return ds_config


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f'throughput averaged with 30 times')
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f'batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}'
        )
        return


def train_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, model_ema=None):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = MyAverageMeter(300)

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        iter_begin_time = time.time()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        model.backward(loss)
        model.step()

        if model_ema is not None:
            model_ema(model)

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(optimizer._global_grad_norm)
        batch_time.update(time.time() - end)
        model_time.update(time.time() - iter_begin_time)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'model_time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f}/{norm_meter.var:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')


@torch.no_grad()
def eval_epoch(config, data_loader, model, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)

        # convert 22k to 1k to evaluate
        if output.size(-1) == 21841:
            convert_file = './meta_data/map22kto1k.txt'
            with open(convert_file, 'r') as f:
                convert_list = [int(line) for line in f.readlines()]
            output = output[:, convert_list]

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                        f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')
    if epoch is not None:
        logger.info(f'[Epoch:{epoch}] * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    else:
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def train(config, ds_config):
    # -------------- build ---------------- #

    _, dataset_val, _, data_loader_train, data_loader_val, _, mixup_fn = build_loader(config)
    model = build_model(config)
    model.cuda()

    if config.MODEL.PRETRAINED:
        load_pretrained(config, model, logger)

    logger.info(ds_config)
    model, optimizer, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=get_parameter_groups(model, config),
        dist_init_required=False,
    )

    try:
        model.register_comm_hook(state=None, hook=fp16_compress_hook)
        logger.info('using fp16_compress_hook!')
    except:
        logger.info('cannot register fp16_compress_hook!')

    model_without_ddp = model.module

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion = build_criterion(config)

    model_ema = None
    if config.TRAIN.EMA.ENABLE:
        model_ema = EMADeepspeed(model, config.TRAIN.EMA.DECAY)

    # -------------- resume ---------------- #

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    client_state = {}
    if config.MODEL.RESUME == '' and config.TRAIN.AUTO_RESUME:
        if os.path.exists(os.path.join(config.OUTPUT, 'latest')):
            config.defrost()
            config.MODEL.RESUME = config.OUTPUT
            config.freeze()
            tag = None
    elif config.MODEL.RESUME:
        config.MODEL.RESUME = os.path.dirname(config.MODEL.RESUME)
        tag = os.path.basename(config.MODEL.RESUME)
    if config.MODEL.RESUME:
        logger.info('loading checkpoint from {}'.format(config.MODEL.RESUME))
        _, client_state = model.load_checkpoint(load_dir=config.MODEL.RESUME, tag=tag)
        logger.info(f'client_state={client_state.keys()}')
        lr_scheduler.load_state_dict(client_state['custom_lr_scheduler'])
        max_accuracy = client_state['max_accuracy']

        if model_ema is not None:
            max_accuracy_ema = client_state.get('max_accuracy_ema', 0.0)
            model_ema.load_state_dict((client_state['model_ema']))

    # -------------- training ---------------- #

    logger.info(f'Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}')
    logger.info(str(model))
    logger.info(get_optimizer_state_str(optimizer))
    logger.info('Start training')
    logger.info('max_accuracy: {}'.format(max_accuracy))
    log_model_statistic(model_without_ddp)

    start_time = time.time()
    start_epoch = client_state['epoch'] + 1 if 'epoch' in client_state else config.TRAIN.START_EPOCH
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        train_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                    model_ema=model_ema)

        if epoch % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS - 1:
            model.save_checkpoint(
                save_dir=config.OUTPUT,
                tag=f'epoch{epoch}',
                client_state={
                    'custom_lr_scheduler': lr_scheduler.state_dict(),
                    'max_accuracy': max_accuracy,
                    'epoch': epoch,
                    'config': config,
                    'max_accuracy_ema': max_accuracy_ema if model_ema is not None else 0.0,
                    'model_ema': model_ema.state_dict() if model_ema is not None else None,
                }
            )

        if epoch % config.EVAL_FREQ == 0:
            acc1, _, _ = eval_epoch(config, data_loader_val, model, epoch)
            logger.info(f'Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%')

            if acc1 > max_accuracy:
                model.save_checkpoint(
                    save_dir=config.OUTPUT,
                    tag='best',
                    client_state={
                        'custom_lr_scheduler': lr_scheduler.state_dict(),
                        'max_accuracy': max_accuracy,
                        'epoch': epoch,
                        'config': config,
                        'max_accuracy_ema': max_accuracy_ema if model_ema is not None else 0.0,
                        'model_ema': model_ema.state_dict() if model_ema is not None else None,
                    }
                )

            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

            if model_ema is not None:
                with model_ema.activate(model):
                    acc1_ema, _, _ = eval_epoch(config, data_loader_val, model, epoch)
                    logger.info(f'[EMA] Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%')
                    max_accuracy_ema = max(max_accuracy_ema, acc1_ema)
                    logger.info(f'[EMA] Max accuracy: {max_accuracy_ema:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def eval(config):
    _, _, _, _, data_loader_val, _, _ = build_loader(config)
    model = build_model(config)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)

    model_wo_ddp = model.module
    if config.MODEL.RESUME:
        try:
            checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
            msg = model_wo_ddp.load_state_dict(checkpoint['model'], strict=False)
            logger.info(msg)
        except:
            try:
                from deepspeed.utils.zero_to_fp32 import \
                    get_fp32_state_dict_from_zero_checkpoint
                ckpt_dir = os.path.dirname(config.MODEL.RESUME)
                tag = os.path.basename(config.MODEL.RESUME)
                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=ckpt_dir, tag=tag)
                model_wo_ddp.load_state_dict(state_dict)
            except:
                checkpoint = torch.load(os.path.join(config.MODEL.RESUME, 'mp_rank_00_model_states.pt'),
                                        map_location='cpu')
                model_wo_ddp.load_state_dict(checkpoint['module'])
    elif config.MODEL.PRETRAINED:
        load_pretrained(config, model_wo_ddp, logger)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)

    eval_epoch(config, data_loader_val, model)


if __name__ == '__main__':
    args, config = parse_option()

    # init distributed env
    if 'SLURM_PROCID' in os.environ:
        print('\nDist init: SLURM')
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        config.defrost()
        config.LOCAL_RANK = gpu
        config.freeze()

        world_size = int(os.environ['SLURM_NTASKS'])
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29501'
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(gpu)
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        os.environ['WORLD_SIZE'] = str(world_size)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         world_size=world_size,
                                         rank=rank)
    torch.distributed.barrier()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(),
                           name=f'{config.MODEL.NAME}')
    logger.info(config.dump())

    if dist.get_rank() == 0:
        save_config(config)
    scale_learning_rate(config, dist.get_world_size())
    seed_everything(config.SEED, dist.get_rank())

    if config.EVAL_MODE:
        eval(config)
    else:
        train(config, build_ds_config(config, args))
