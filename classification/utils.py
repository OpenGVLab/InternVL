# --------------------------------------------------------
# InternVL
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import math
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from timm.utils import get_state_dict

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_ema_checkpoint(config, model_ema, logger):
    logger.info(
        f'==============> Resuming form {config.MODEL.RESUME}....................'
    )
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(config.MODEL.RESUME,
                                                        map_location='cpu',
                                                        check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    assert isinstance(checkpoint, dict)
    if 'model_ema' in checkpoint:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_ema'].items():
            if model_ema.ema_has_module:
                name = 'module.' + k if not k.startswith('module') else k
            else:
                name = k
            new_state_dict[name] = v
        msg = model_ema.ema.load_state_dict(new_state_dict, strict=False)
        logger.info(msg)
        logger.info('Loaded state_dict_ema')
    else:
        logger.warning(
            'Failed to find state_dict_ema, starting from loaded model weights'
        )

    max_accuracy_ema = 0
    if 'max_accuracy_ema' in checkpoint:
        max_accuracy_ema = checkpoint['max_accuracy_ema']
    if 'ema_decay' in checkpoint:
        model_ema.decay = checkpoint['ema_decay']
    return max_accuracy_ema


def load_checkpoint(config, model, optimizer, lr_scheduler, scaler, logger):
    logger.info(
        f'==============> Resuming form {config.MODEL.RESUME}....................'
    )
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(config.MODEL.RESUME,
                                                        map_location='cpu',
                                                        check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    print('resuming model')

    model_checkpoint = checkpoint['model']
    msg = model.load_state_dict(model_checkpoint, strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if optimizer is not None:
            print('resuming optimizer')
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('resume optimizer failed')
        if lr_scheduler is not None:
            print('resuming lr_scheduler')
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != 'O0' and checkpoint['config'].AMP_OPT_LEVEL != 'O0':
            scaler.load_state_dict(checkpoint['amp'])
        logger.info(
            f"=> loaded successfully {config.MODEL.RESUME} (epoch {checkpoint['epoch']})"
        )
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()

    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(
        f'==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......'
    )
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    state_dict = checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']

    first_key = list(state_dict.keys())[0]
    # delete teacher weights
    if 'student' in first_key or 'teacher' in first_key:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'student_proj' in k:
                continue
            if 'student' in k:
                new_k = k.replace('student.', '')
                new_state_dict[new_k] = v
        state_dict = new_state_dict

    # weights from sim
    if 'mask_token' in first_key:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'mm_dcnv3' in k:
                continue
            if 'dcnv3' not in k and 'clip_projector' not in k:
                continue
            new_k = k.replace('dcnv3.', '')
            new_state_dict[new_k] = v
        new_state_dict['fc_norm.weight'] = state_dict[
            'clip.classifier_ln.weight']
        new_state_dict['fc_norm.bias'] = state_dict['clip.classifier_ln.bias']
        new_state_dict['head.weight'] = state_dict['clip.classifier.weight']
        new_state_dict['head.bias'] = state_dict['clip.classifier.bias']
        state_dict = new_state_dict

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if 'relative_position_index' in k
    ]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if 'relative_coords_table' in k
    ]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if 'attn_mask' in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if 'relative_position_bias_table' in k
    ]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f'Error in loading {k}, passing......')
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                    size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if 'absolute_pos_embed' in k
    ]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f'Error in loading {k}, passing......')
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained,
                    size=(S2, S2),
                    mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    if 'head.bias' in state_dict:
        head_bias_pretrained = state_dict['head.bias']
        Nc1 = head_bias_pretrained.shape[0]
        Nc2 = model.head.bias.shape[0]

        if (Nc1 != Nc2):
            if config.TRAIN.RAND_INIT_FT_HEAD:
                model.head.weight.data = model.head.weight.data * 0.001
                model.head.bias.data = model.head.bias.data * 0.001
                del state_dict['head.weight']
                del state_dict['head.bias']
                logger.warning(f'Error in loading classifier head, re-init classifier head to 0')
            elif Nc1 == 21841 and Nc2 == 1000:
                logger.info('loading ImageNet-22K weight to ImageNet-1K ......')
                map22kto1k_path = 'meta_data/map22kto1k.txt'
                logger.info(map22kto1k_path)
                with open(map22kto1k_path) as f:
                    map22kto1k = f.readlines()
                map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f'=> loaded successfully {config.MODEL.PRETRAINED}')

    del checkpoint
    torch.cuda.empty_cache()


def convert_22k_head_to_1k(model, logger):
    head_weight = model.module.head.weight
    head_bias = model.module.head.bias
    Nc1 = head_bias.shape[0]

    if Nc1 == 21841:
        logger.info('converting ImageNet-22K head to ImageNet-1K ......')
        map22kto1k_path = 'meta_data/map22kto1k.txt'
        logger.info(map22kto1k_path)
        with open(map22kto1k_path) as f:
            map22kto1k = f.readlines()
        map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
        model.module.head.weight = torch.nn.Parameter(head_weight[map22kto1k, :])
        model.module.head.bias = torch.nn.Parameter(head_bias[map22kto1k])
    else:
        logger.warning(f'Error in converting classifier head')

    return model


def save_checkpoint(config,
                    epoch,
                    model,
                    max_accuracy,
                    optimizer,
                    lr_scheduler,
                    scaler,
                    logger,
                    model_ema=None,
                    max_accuracy_ema=None,
                    ema_decay=None,
                    model_ems=None,
                    max_accuracy_ems=None,
                    ems_model_num=None,
                    best=None):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'max_accuracy': max_accuracy,
        'epoch': epoch,
        'config': config
    }
    if model_ema is not None:
        save_state['model_ema'] = get_state_dict(model_ema)
    if max_accuracy_ema is not None:
        save_state['max_accuracy_ema'] = max_accuracy_ema
    if ema_decay is not None:
        save_state['ema_decay'] = ema_decay
    if model_ems is not None:
        save_state['model_ems'] = get_state_dict(model_ems)
    if max_accuracy_ems is not None:
        save_state['max_accuracy_ems'] = max_accuracy_ems
    if ems_model_num is not None:
        save_state['ems_model_num'] = ems_model_num
    if config.AMP_OPT_LEVEL != 'O0':
        # save_state['amp'] = amp.state_dict()
        save_state['amp'] = scaler.state_dict()
    if best is None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{best}.pth')
    logger.info(f'{save_path} saving......')
    torch.save(save_state, save_path)
    logger.info(f'{save_path} saved !!!')

    if dist.get_rank() == 0 and isinstance(epoch, int):
        to_del = epoch - config.SAVE_CKPT_NUM * config.SAVE_FREQ
        old_ckpt = os.path.join(config.OUTPUT, f'ckpt_epoch_{to_del}.pth')
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f'All checkpoints founded in {output_dir}: {checkpoints}')
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints],
            key=os.path.getmtime)
        print(f'The latest checkpoint founded: {latest_checkpoint}')
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


# https://github.com/facebookresearch/ConvNeXt/blob/main/utils.py
class NativeScalerWithGradNormCount:
    state_dict_key = 'amp_scaler'

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self,
                 loss,
                 optimizer,
                 clip_grad=None,
                 parameters=None,
                 create_graph=False,
                 update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class MyAverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, max_len=-1):
        self.val_list = []
        self.count = []
        self.max_len = max_len
        self.val = 0
        self.avg = 0
        self.var = 0

    def update(self, val):
        self.val = val
        self.avg = 0
        self.var = 0
        if not math.isnan(val) and not math.isinf(val):
            self.val_list.append(val)
        if self.max_len > 0 and len(self.val_list) > self.max_len:
            self.val_list = self.val_list[-self.max_len:]
        if len(self.val_list) > 0:
            self.avg = np.mean(np.array(self.val_list))
            self.var = np.std(np.array(self.val_list))
