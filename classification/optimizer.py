# --------------------------------------------------------
# InternVL
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from torch import optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
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

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    use_zero = config.TRAIN.OPTIMIZER.USE_ZERO
    if use_zero:
        print(f'\nUse Zero!')
        if opt_lower == 'sgd':
            # an ugly implementation
            # this problem is fixed after torch 1.12
            # https://github.com/pytorch/pytorch/issues/71347

            # before 1.12, we could only pass list to zero optimizer, so we first pass parameters[0] with its lr and weight decay,
            # then we add other parameter via parameter group.

            optimizer = ZeroRedundancyOptimizer(
                parameters[0]['params'],
                optimizer_class=optim.SGD,
                momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                lr=parameters[0]['lr'], weight_decay=parameters[0]['weight_decay']
            )
            if len(parameters) > 1:
                for param_group in parameters[1:]:
                    optimizer.add_param_group(param_group)
        elif opt_lower == 'adamw':
            optimizer = ZeroRedundancyOptimizer(
                parameters[0]['params'],
                optimizer_class=optim.AdamW,
                eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                lr=parameters[0]['lr'], weight_decay=parameters[0]['weight_decay']
            )
            if len(parameters) > 1:
                for param_group in parameters[1:]:
                    optimizer.add_param_group(param_group)
    else:
        if opt_lower == 'sgd':
            optimizer = optim.SGD(parameters,
                                  momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                                  nesterov=True,
                                  lr=config.TRAIN.BASE_LR,
                                  weight_decay=config.TRAIN.WEIGHT_DECAY)
        elif opt_lower == 'sgd_linear_probing':
            optimizer = optim.SGD(parameters,
                                  momentum=0.9,
                                  nesterov=False,
                                  lr=config.TRAIN.BASE_LR,
                                  weight_decay=0)
        elif opt_lower == 'adamw':
            optimizer = optim.AdamW(parameters,
                                    eps=config.TRAIN.OPTIMIZER.EPS,
                                    betas=config.TRAIN.OPTIMIZER.BETAS,
                                    lr=config.TRAIN.BASE_LR,
                                    weight_decay=config.TRAIN.WEIGHT_DECAY)
        else:
            raise NotImplementedError
    return optimizer


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def check_keywords_in_dict(name, keywords_dict):
    for k, v in keywords_dict.items():
        if k in name:
            return v
    return None


def set_weight_decay_and_lr(
        model,
        weight_decay,
        base_lr,
        skip_list=(),
        skip_keywords=(),
        lr_layer_decay=None,
        lr_layer_decay_ratio=None,
        freeze_backbone=None,
        dcn_lr_mul=None,
        layerwise_lr=True,
):
    parameters = []
    no_decay_name = []
    lr_ratio_log = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if freeze_backbone:
            for i in freeze_backbone:
                if f'levels.{i}' in name:
                    param.requires_grad = False
        # 1. check wd
        if len(param.shape) == 1 or name.endswith('.bias') or (
                name in skip_list) or check_keywords_in_name(name, skip_keywords):
            wd = 0.
            no_decay_name.append(name)
        else:
            wd = weight_decay

        if lr_layer_decay:
            print('layer-wise lr decay is used !')
            assert hasattr(model, 'lr_decay_keywords')
            lr_ratio_keywards = model.lr_decay_keywords(lr_layer_decay_ratio)

            # 2. check lr
            ratio = check_keywords_in_dict(name, lr_ratio_keywards)
            if ratio is not None:
                lr = ratio * base_lr
            else:
                lr = base_lr

            # dcn lr
            if dcn_lr_mul is not None:
                if 'offset' in name or 'attention_weights' in name or 'center_feature_scale_proj' in name or 'alpha_beta' in name:
                    lr = dcn_lr_mul * lr

            lr_ratio_log[name] = (base_lr, ratio, wd, param.requires_grad)
        else:
            lr = base_lr
        parameters.append({'params': [param], 'weight_decay': wd, 'lr': lr, 'name': name})

    print('no decay params: {no_decay_name}')
    if layerwise_lr:
        print('lr_ratio_params:')
        for k, v in lr_ratio_log.items():
            print(k, v)

    return parameters
