# --------------------------------------------------------
# InternVL
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Load data to memory
_C.DATA.IMG_ON_MEMORY = False
# Name of the build_transform function
_C.DATA.TRANSFORM = 'build_transform'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'intern_vit_6b'
# Model name
_C.MODEL.NAME = 'intern_vit_6b'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Drop path type
_C.MODEL.DROP_PATH_TYPE = 'linear'  # linear, uniform
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# INTERN_VIT_6B parameters
_C.MODEL.INTERN_VIT_6B = CN()
_C.MODEL.INTERN_VIT_6B.PATCH_SIZE = 14
_C.MODEL.INTERN_VIT_6B.PRETRAIN_SIZE = 224
_C.MODEL.INTERN_VIT_6B.QKV_BIAS = False
_C.MODEL.INTERN_VIT_6B.EMBED_DIM = 3200
_C.MODEL.INTERN_VIT_6B.NUM_HEADS = 25
_C.MODEL.INTERN_VIT_6B.MLP_RATIO = 4
_C.MODEL.INTERN_VIT_6B.INIT_VALUES = 0.1
_C.MODEL.INTERN_VIT_6B.QK_NORMALIZATION = True
_C.MODEL.INTERN_VIT_6B.DEPTH = 48
_C.MODEL.INTERN_VIT_6B.USE_FLASH_ATTN = True
_C.MODEL.INTERN_VIT_6B.FREEZE_VIT = True
_C.MODEL.INTERN_VIT_6B.PRETRAINED = None
_C.MODEL.INTERN_VIT_6B.CLS_TARGET = 'cls_patch_concat'
_C.MODEL.INTERN_VIT_6B.HEAD_NORM_TYPE = 'bn'

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# ZeRO
_C.TRAIN.OPTIMIZER.USE_ZERO = False
# freeze backbone
_C.TRAIN.OPTIMIZER.FREEZE_BACKBONE = None
# dcn lr
_C.TRAIN.OPTIMIZER.DCN_LR_MUL = None

# EMA
_C.TRAIN.EMA = CN()
_C.TRAIN.EMA.ENABLE = False
_C.TRAIN.EMA.DECAY = 0.9998

# LR_LAYER_DECAY
_C.TRAIN.LR_LAYER_DECAY = False
_C.TRAIN.LR_LAYER_DECAY_RATIO = 0.875

# FT head init weights
_C.TRAIN.RAND_INIT_FT_HEAD = False

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'
# RandomResizedCrop
_C.AUG.RANDOM_RESIZED_CROP = False
_C.AUG.MEAN = (0.485, 0.456, 0.406)
_C.AUG.STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# eval freq
_C.EVAL_FREQ = 1
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
_C.EVAL_22K_TO_1K = False

_C.AMP_TYPE = 'float16'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if hasattr(args, 'opts') and args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if hasattr(args, 'batch_size') and args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if hasattr(args, 'dataset') and args.dataset:
        config.DATA.DATASET = args.dataset
    if hasattr(args, 'data_path') and args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if hasattr(args, 'zip') and args.zip:
        config.DATA.ZIP_MODE = True
    if hasattr(args, 'cache_mode') and args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if hasattr(args, 'pretrained') and args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if hasattr(args, 'resume') and args.resume:
        config.MODEL.RESUME = args.resume
    if hasattr(args, 'accumulation_steps') and args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if hasattr(args, 'use_checkpoint') and args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if hasattr(args, 'amp_opt_level') and args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if hasattr(args, 'output') and args.output:
        config.OUTPUT = args.output
    if hasattr(args, 'tag') and args.tag:
        config.TAG = args.tag
    if hasattr(args, 'eval') and args.eval:
        config.EVAL_MODE = True
    if hasattr(args, 'throughput') and args.throughput:
        config.THROUGHPUT_MODE = True
    if hasattr(args, 'save_ckpt_num') and args.save_ckpt_num:
        config.SAVE_CKPT_NUM = args.save_ckpt_num
    if hasattr(args, 'use_zero') and args.use_zero:
        config.TRAIN.OPTIMIZER.USE_ZERO = True
    # set local rank for distributed training
    if hasattr(args, 'local_rank') and args.local_rank:
        config.LOCAL_RANK = args.local_rank

    # output folder
    config.MODEL.NAME = args.cfg.split('/')[-1].replace('.yaml', '')
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME)
    # config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
