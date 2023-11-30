# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import datetime
import os
import random
import subprocess
import time
from contextlib import suppress

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from config import get_config
from dataset import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ApexScaler, AverageMeter, ModelEma, accuracy
from utils import MyAverageMeter
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import (auto_resume_helper, get_grad_norm, load_checkpoint,
                   load_ema_checkpoint, load_pretrained, reduce_tensor,
                   save_checkpoint)

try:
    from apex import amp

    has_apex = True
except ImportError:
    has_apex = False
# assert not has_apex, "The code is modified based on native amp"

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


def obsolete_torch_version(torch_version, version_threshold):
    return torch_version == 'parrots' or torch_version <= version_threshold


def parse_option():
    parser = argparse.ArgumentParser(
        'InternVL training and evaluation script', add_help=False)
    parser.add_argument('--cfg',
                        type=str,
                        required=True,
                        metavar='FILE',
                        help='path to config file')
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')

    # easy config modification
    parser.add_argument('--batch-size',
                        type=int,
                        help='batch size for single GPU')
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset name',
                        default=None)
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip',
                        action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument(
        '--cache-mode',
        type=str,
        default='part',
        choices=['no', 'full', 'part'],
        help='no: no cache, '
             'full: cache all data, '
             'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
    )
    parser.add_argument(
        '--pretrained',
        help=
        'pretrained weight from checkpoint, could be imagenet22k pretrained weight'
    )
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps',
                        type=int,
                        default=1,
                        help='gradient accumulation steps')
    parser.add_argument(
        '--use-checkpoint',
        action='store_true',
        help='whether to use gradient checkpointing to save memory')
    parser.add_argument(
        '--amp-opt-level',
        type=str,
        default='O1',
        choices=['O0', 'O1', 'O2'],
        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument(
        '--output',
        default='work_dirs',
        type=str,
        metavar='PATH',
        help=
        'root of output folder, the full path is <output>/<model_name>/<tag> (default: output)'
    )
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval',
                        action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput',
                        action='store_true',
                        help='Test throughput only')
    parser.add_argument('--save-ckpt-num', default=1, type=int)
    parser.add_argument(
        '--use-zero',
        action='store_true',
        help='whether to use ZeroRedundancyOptimizer (ZeRO) to save memory')

    # distributed training
    parser.add_argument('--local-rank',
                        type=int,
                        required=True,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--launcher',
                        choices=['pytorch', 'slurm'],
                        default='pytorch')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


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


def main(config):
    # prepare data loaders
    dataset_train, dataset_val, dataset_test, data_loader_train, \
    data_loader_val, data_loader_test, mixup_fn = build_loader(config)

    # build runner
    logger.info(f'Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}')
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    # build optimizer
    optimizer = build_optimizer(config, model)

    if config.AMP_OPT_LEVEL != 'O0':
        config.defrost()
        if has_native_amp:
            config.native_amp = True
            use_amp = 'native'
        elif has_apex:
            config.apex_amp = True
            use_amp = 'apex'
        else:
            use_amp = None
            logger.warning(
                'Neither APEX or native Torch AMP is available, using float32. '
                'Install NVIDA apex or upgrade to PyTorch 1.6')
        config.freeze()

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if config.AMP_OPT_LEVEL != 'O0':
        if use_amp == 'apex':
            model, optimizer = amp.initialize(model,
                                              optimizer,
                                              opt_level=config.AMP_OPT_LEVEL)
            loss_scaler = ApexScaler()
            if config.LOCAL_RANK == 0:
                logger.info(
                    'Using NVIDIA APEX AMP. Training in mixed precision.')
        if use_amp == 'native':
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()
            if config.LOCAL_RANK == 0:
                logger.info(
                    'Using native Torch AMP. Training in mixed precision.')
        else:
            if config.LOCAL_RANK == 0:
                logger.info('AMP not enabled. Training in float32.')

    # put model on gpus
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)

    # try:
    #     model.register_comm_hook(state=None, hook=fp16_compress_hook)
    #     logger.info('using fp16_compress_hook!')
    # except:
    #     logger.info("cannot register fp16_compress_hook!")

    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f'number of GFLOPs: {flops / 1e9}')

    # build learning rate scheduler
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train)) \
        if not config.EVAL_MODE else None

    # build criterion
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_ema_accuracy = 0.0
    # set auto resume
    if config.MODEL.RESUME == '' and config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f'auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}'
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume'
            )

    # set resume and pretrain
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer,
                                       lr_scheduler, loss_scaler, logger)

        if data_loader_val is not None:
            if config.DATA.DATASET == 'imagenet-real':
                filenames = dataset_val.filenames()
                filenames = [os.path.basename(item) for item in filenames]
                from dataset.imagenet_real import RealLabelsImagenet
                real_labels = RealLabelsImagenet(filenames, real_json='meta_data/real.json')
                acc1, acc5, loss = validate_real(config, data_loader_val, model, real_labels, amp_autocast=amp_autocast)
                logger.info(
                    f'ReaL Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%'
                )
            else:
                acc1, acc5, loss = validate(config, data_loader_val, model, amp_autocast=amp_autocast)
                logger.info(
                    f'Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%'
                )
    elif config.MODEL.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger)
        if data_loader_val is not None:
            acc1, acc5, loss = validate(config, data_loader_val, model, amp_autocast=amp_autocast)
            logger.info(
                f'Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%'
            )

    # evaluate EMA
    model_ema = None
    if config.TRAIN.EMA.ENABLE:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(model, decay=config.TRAIN.EMA.DECAY)
        print('Using EMA with decay = %.8f' % config.TRAIN.EMA.DECAY)
        if config.MODEL.RESUME:
            load_ema_checkpoint(config, model_ema, logger)
            if config.DATA.DATASET == 'imagenet-real':
                # assert only one gpu
                assert dist.get_world_size() == 1, 'imagenet-real should test with one gpu'
                filenames = dataset_val.filenames()
                filenames = [os.path.basename(item) for item in filenames]
                from dataset.imagenet_real import RealLabelsImagenet
                real_labels = RealLabelsImagenet(filenames, real_json='meta_data/real.json')
                acc1, acc5, loss = validate_real(config, data_loader_val, model_ema.ema, real_labels,
                                                 amp_autocast=amp_autocast)
                logger.info(
                    f'ReaL Accuracy of the ema network on the {len(dataset_val)} test images: {acc1:.1f}%'
                )
            else:
                acc1, acc5, loss = validate(config, data_loader_val, model_ema.ema, amp_autocast=amp_autocast)
                logger.info(
                    f'Accuracy of the ema network on the {len(dataset_val)} test images: {acc1:.1f}%'
                )

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)

    if config.EVAL_MODE:
        return

    # train
    logger.info('Start training')
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config,
                        model,
                        criterion,
                        data_loader_train,
                        optimizer,
                        epoch,
                        mixup_fn,
                        lr_scheduler,
                        amp_autocast,
                        loss_scaler,
                        model_ema=model_ema)
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)) and config.TRAIN.OPTIMIZER.USE_ZERO:
            optimizer.consolidate_state_dict(to=0)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config,
                            epoch,
                            model_without_ddp,
                            max_accuracy,
                            optimizer,
                            lr_scheduler,
                            loss_scaler,
                            logger,
                            model_ema=model_ema)
        if data_loader_val is not None and epoch % config.EVAL_FREQ == 0:
            acc1, acc5, loss = validate(config, data_loader_val, model, epoch, amp_autocast=amp_autocast)
            logger.info(
                f'Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%'
            )
            if dist.get_rank() == 0 and acc1 > max_accuracy:
                save_checkpoint(config,
                                epoch,
                                model_without_ddp,
                                max_accuracy,
                                optimizer,
                                lr_scheduler,
                                loss_scaler,
                                logger,
                                model_ema=model_ema,
                                best='best')
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

            if config.TRAIN.EMA.ENABLE:
                acc1, acc5, loss = validate(config, data_loader_val,
                                            model_ema.ema, epoch, amp_autocast=amp_autocast)
                logger.info(
                    f'Accuracy of the ema network on the {len(dataset_val)} test images: {acc1:.1f}%'
                )
                if dist.get_rank() == 0 and acc1 > max_ema_accuracy:
                    save_checkpoint(config,
                                    epoch,
                                    model_without_ddp,
                                    max_accuracy,
                                    optimizer,
                                    lr_scheduler,
                                    loss_scaler,
                                    logger,
                                    model_ema=model_ema,
                                    best='ema_best')
                max_ema_accuracy = max(max_ema_accuracy, acc1)
                logger.info(f'Max ema accuracy: {max_ema_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config,
                    model,
                    criterion,
                    data_loader,
                    optimizer,
                    epoch,
                    mixup_fn,
                    lr_scheduler,
                    amp_autocast=suppress,
                    loss_scaler=None,
                    model_ema=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = MyAverageMeter(300)

    start = time.time()
    end = time.time()

    amp_type = torch.float16 if config.AMP_TYPE == 'float16' else torch.bfloat16
    for idx, (samples, targets) in enumerate(data_loader):
        iter_begin_time = time.time()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if not obsolete_torch_version(TORCH_VERSION,
                                      (1, 9)) and config.AMP_OPT_LEVEL != 'O0':
            with amp_autocast(dtype=amp_type):
                outputs = model(samples)
        else:
            with amp_autocast():
                outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if not obsolete_torch_version(
                    TORCH_VERSION, (1, 9)) and config.AMP_OPT_LEVEL != 'O0':
                with amp_autocast(dtype=amp_type):
                    loss = criterion(outputs, targets)
                    loss = loss / config.TRAIN.ACCUMULATION_STEPS
            else:
                with amp_autocast():
                    loss = criterion(outputs, targets)
                    loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != 'O0':
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(loss,
                                        optimizer,
                                        clip_grad=config.TRAIN.CLIP_GRAD,
                                        parameters=model.parameters(),
                                        create_graph=is_second_order,
                                        update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            if not obsolete_torch_version(
                    TORCH_VERSION, (1, 9)) and config.AMP_OPT_LEVEL != 'O0':
                with amp_autocast(dtype=amp_type):
                    loss = criterion(outputs, targets)
            else:
                with amp_autocast():
                    loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != 'O0':
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(loss,
                                        optimizer,
                                        clip_grad=config.TRAIN.CLIP_GRAD,
                                        parameters=model.parameters(),
                                        create_graph=is_second_order,
                                        update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
                if model_ema is not None:
                    model_ema.update(model)
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
                if model_ema is not None:
                    model_ema.update(model)

            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:
            norm_meter.update(grad_norm.item())
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
    logger.info(
        f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}'
    )


@torch.no_grad()
def validate_real(config, data_loader, model, real_labels, amp_autocast=suppress):
    # https://github.com/baaivision/EVA/blob/master/EVA-01/eva/engine_for_finetuning.py#L195
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    amp_type = torch.float16 if config.AMP_TYPE == 'float16' else torch.bfloat16
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if not obsolete_torch_version(TORCH_VERSION, (1, 9)) and config.AMP_OPT_LEVEL != 'O0':
            with amp_autocast(dtype=amp_type):
                output = model(images)
        else:
            with amp_autocast():
                output = model(images)

        # convert 22k to 1k to evaluate
        if output.size(-1) == 21841:
            convert_file = './meta_data/map22kto1k.txt'
            with open(convert_file, 'r') as f:
                convert_list = [int(line) for line in f.readlines()]
            output = output[:, convert_list]

        real_labels.add_result(output)

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

    # real labels mode replaces topk values at the end
    top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)

    print('* ReaL Acc@1 {:.3f} Acc@5 {:.3f} loss {losses:.3f}'
          .format(top1a, top5a, losses=loss_meter.avg))

    return top1a, top5a, loss_meter.avg


@torch.no_grad()
def validate(config, data_loader, model, epoch=None, amp_autocast=suppress):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    amp_type = torch.float16 if config.AMP_TYPE == 'float16' else torch.bfloat16
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if not obsolete_torch_version(TORCH_VERSION, (1, 9)) and config.AMP_OPT_LEVEL != 'O0':
            with amp_autocast(dtype=amp_type):
                output = model(images)
        else:
            with amp_autocast():
                output = model(images)

        # convert 22k to 1k to evaluate
        if output.size(-1) == 21841:
            convert_file = './meta_data/map22kto1k.txt'
            with open(convert_file, 'r') as f:
                convert_list = [int(line) for line in f.readlines()]
            output = output[:, convert_list]

        if config.DATA.DATASET == 'imagenet_a':
            from dataset.imagenet_a_r_indices import imagenet_a_mask
            output = output[:, imagenet_a_mask]
        elif config.DATA.DATASET == 'imagenet_r':
            from dataset.imagenet_a_r_indices import imagenet_r_mask
            output = output[:, imagenet_r_mask]

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
        logger.info(
            f'[Epoch:{epoch}] * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}'
        )
    else:
        logger.info(
            f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != 'O0':
        assert has_native_amp, 'Please update pytorch(1.6+) to support amp!'

    # init distributed env
    if _.launcher == 'slurm':
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

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    print(config.AMP_OPT_LEVEL, _.amp_opt_level)

    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(),
                           name=f'{config.MODEL.NAME}')

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, 'config.json')
        with open(path, 'w') as f:
            f.write(config.dump())
        logger.info(f'Full config saved to {path}')

    # print config
    logger.info(config.dump())

    main(config)
