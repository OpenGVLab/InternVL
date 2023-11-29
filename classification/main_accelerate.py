# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import datetime
import argparse
import os
import time
import logging
import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from accelerate import Accelerator
from accelerate import GradScalerKwargs
from accelerate.logging import get_logger
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy, ModelEma
from tqdm import tqdm
import warnings

from config import get_config
from models import build_model
from dataset import build_loader2
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from utils import load_pretrained, load_ema_checkpoint
from ddp_hooks import fp16_compress_hook

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


def parse_option():
    parser = argparse.ArgumentParser(
        'InternImage training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--dataset', type=str, help='dataset name', default=None)
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                        'full: cache all data, '
                        'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
                        )
    parser.add_argument('--pretrained', help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)'
                        )
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--save-ckpt-num', default=1, type=int)
    parser.add_argument('--accumulation-steps', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--disable-grad-scalar', action='store_true', help='disable Grad Scalar')
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    config.defrost()
    config.TRAIN.OPTIMIZER.USE_ZERO = False
    config.OUTPUT += '_deepspeed'
    config.DATA.IMG_ON_MEMORY = False
    config.freeze()
    return args, config


def seed_everything(seed, rank):
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def save_config(config):
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")


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
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * num_processes / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * num_processes / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * num_processes / 512.0
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


def setup_autoresume(config):
    if config.MODEL.RESUME == '' and config.TRAIN.AUTO_RESUME:
        last_checkpoint = os.path.join(config.OUTPUT, 'last')
        resume_file = last_checkpoint if os.path.exists(last_checkpoint) else None

        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')


def load_model_checkpoint(config, model, accelerator):
    if config.MODEL.RESUME:
        try:
            checkpoint = torch.load(config.MODEL.RESUME)['model']
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        except:
            accelerator.load_state(config.MODEL.RESUME)
    elif config.MODEL.PRETRAINED:
        try:
            load_pretrained(config, model, logger)
        except:
            accelerator.load_state(config.MODEL.PRETRAINED)
    return model


def save_checkpoint(save_dir, accelerator, epoch, max_acc, config, lr_scheduler=None):
    # let accelerator handle the model and optimizer state for ddp and deepspeed.
    accelerator.save_state(save_dir)

    if accelerator.is_main_process:
        save_state = {
            'lr_scheduler': lr_scheduler.state_dict(),
            'max_acc': max_acc,
            'epoch': epoch,
            'config': config
        }
        torch.save(save_state, os.path.join(save_dir, 'additional_state.pth'))


def load_checkpoint_if_needed(accelerator, config, lr_scheduler=None):
    setup_autoresume(config)
    save_dir = config.MODEL.RESUME
    if not save_dir:
        return 0.0
    accelerator.load_state(save_dir)
    checkpoint = torch.load(os.path.join(save_dir, 'additional_state.pth'), map_location='cpu')
    if lr_scheduler is not None:
        logger.info('resuming lr_scheduler')
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    config.defrost()
    config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
    config.freeze()
    max_acc = checkpoint.get('max_acc', 0.0)
    logger.info(f"=> loaded successfully {config.MODEL.RESUME} (epoch {checkpoint['epoch']})")
    return max_acc


def log_model_statistic(model_wo_ddp):
    n_parameters = sum(p.numel() for p in model_wo_ddp.parameters()
                       if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_wo_ddp, 'flops'):
        flops = model_wo_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")


def train_epoch(*, model, optimizer, data_loader, scheduler, criterion, mixup_fn,
                accelerator: Accelerator, epoch, config):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()

    gradient_accumulation_steps = config.TRAIN.ACCUMULATION_STEPS

    for step, (samples, targets) in enumerate(data_loader):
        iter_begin_time = time.time()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with accelerator.accumulate(model):
            outputs = model(samples)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            optimizer.step()
            optimizer.zero_grad()

        accelerator.wait_for_everyone()

        if (step + 1) % gradient_accumulation_steps == 0:
            if scheduler is not None:
                scheduler.step_update((epoch * num_steps + step) // gradient_accumulation_steps)

            batch_time.update(time.time() - end)
            model_time.update(time.time() - iter_begin_time)
            loss_meter.update(loss.item())
            end = time.time()

        if accelerator.is_main_process and step % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - step)

            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{step}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.10f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'model_time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.8f} ({loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')


@torch.no_grad()
def eval_epoch(*, config, data_loader, model, accelerator: Accelerator):
    model.eval()

    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for idx, (images, target) in enumerate(tqdm(data_loader, disable=accelerator.is_main_process)):
        output = model(images)

        # convert 22k to 1k to evaluate
        if output.size(-1) == 21841:
            convert_file = './meta_data/map22kto1k.txt'
            with open(convert_file, 'r') as f:
                convert_list = [int(line) for line in f.readlines()]
            output = output[:, convert_list]

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accelerator.gather(acc1).mean(0)
        acc5 = accelerator.gather(acc5).mean(0)

        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        if (idx + 1) % config.PRINT_FREQ == 0 or idx + 1 == len(data_loader):
            logger.info(f'Test: [{idx+1}/{len(data_loader)}]\t'
                        f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                        f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                        )
    return acc1_meter.avg


def eval(config, accelerator: Accelerator):
    _, _, _, _, validate_dataloader, _, _ = build_loader2(config)
    model = build_model(config)
    model, validate_dataloader = accelerator.prepare(model, validate_dataloader)
    model = load_model_checkpoint(config, model, accelerator)
    log_model_statistic(accelerator.unwrap_model(model))
    eval_epoch(config=config, data_loader=validate_dataloader, model=model, accelerator=accelerator)


def train(config, accelerator: Accelerator):
    _, _, _, training_dataloader, validate_dataloader, _, mixup_fn = build_loader2(config)
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    criterion = build_criterion(config)

    model, optimizer, training_dataloader, validate_dataloader = accelerator.prepare(
        model, optimizer, training_dataloader, validate_dataloader)

    effective_update_steps_per_epoch = len(training_dataloader) // config.TRAIN.ACCUMULATION_STEPS
    lr_scheduler = build_scheduler(config, optimizer, effective_update_steps_per_epoch)

    try:
        model.register_comm_hook(state=None, hook=fp16_compress_hook)
        logger.info('using fp16_compress_hook!')
    except:
        logger.info("cannot register fp16_compress_hook!")

    max_acc = load_checkpoint_if_needed(accelerator, config, lr_scheduler)

    logger.info(f"Created model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    logger.info(str(model))
    logger.info("Effective Optimizer Steps: {}".format(effective_update_steps_per_epoch))
    logger.info("Start training")
    logger.info("Max accuracy: {}".format(max_acc))
    log_model_statistic(accelerator.unwrap_model(model))

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_epoch(model=model, optimizer=optimizer, data_loader=training_dataloader,
                    scheduler=lr_scheduler, criterion=criterion, mixup_fn=mixup_fn,
                    accelerator=accelerator, epoch=epoch, config=config)
        acc = eval_epoch(config=config, data_loader=validate_dataloader, model=model,
                         accelerator=accelerator)

        accelerator.wait_for_everyone()
        if acc > max_acc:
            max_acc = acc
            save_checkpoint(os.path.join(config.OUTPUT, 'best'), accelerator, epoch, max_acc, config, lr_scheduler)
        logger.info(f'Max Acc@1 {max_acc:.3f}')
        save_checkpoint(os.path.join(config.OUTPUT, 'last'), accelerator, epoch, max_acc, config, lr_scheduler)


def main():
    args, config = parse_option()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        filename=os.path.join(config.OUTPUT, 'run.log'),
        level=logging.INFO,
    )

    loggers = ['tensorboard']
    accelerator = Accelerator(
        log_with=loggers,
        project_dir=config.OUTPUT,
        gradient_accumulation_steps=config.TRAIN.ACCUMULATION_STEPS,
        # When use deepspeed, you could not comment this out
        # even if you set loss scale to 1.0 in deepspeed config.
        kwargs_handlers=[GradScalerKwargs(enabled=not args.disable_grad_scalar)],
    )
    logger.info(accelerator.state, main_process_only=False)

    scale_learning_rate(config, accelerator.num_processes)
    seed_everything(config.SEED, accelerator.process_index)
    save_config(config)

    logger.info(config.dump())

    if config.EVAL_MODE:
        eval(config, accelerator)
    else:
        train(config, accelerator)


if __name__ == '__main__':
    main()
