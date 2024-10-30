# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import time

import torch
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string
from models.intern_vit_6b import InternViT6B
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('config', nargs='?', type=str, default=None)
args = parser.parse_args()

configs = {
    'a': {
        'embed_dim': 3968,
        'num_heads': 62,
        'mlp_ratio': 4,
        'depth': 32
    },
    'e': {
        'embed_dim': 3200,
        'num_heads': 50,
        'mlp_ratio': 4,
        'depth': 48
    },
    'f': {
        'embed_dim': 3200,
        'num_heads': 25,
        'mlp_ratio': 4,
        'depth': 48
    },
    'g': {
        'embed_dim': 2496,
        'num_heads': 39,
        'mlp_ratio': 8,
        'depth': 48
    },
    'i': {
        'embed_dim': 2816,
        'num_heads': 44,
        'mlp_ratio': 4,
        'depth': 64
    },
    'm': {
        'embed_dim': 2496,
        'num_heads': 39,
        'mlp_ratio': 4,
        'depth': 80
    },
}


def sa_flops(h, w, dim):
    return 2 * h * w * h * w * dim


def get_flops(model, input_shape):
    flops, params = get_model_complexity_info(model,
                                              input_shape,
                                              as_strings=False)
    _, H, W = input_shape
    print(flops, params)
    for i in range(model.depth):
        flops += sa_flops(H // model.patch_size, W // model.patch_size,
                          model.embed_dim)
    return flops_to_string(flops), params_to_string(params)


if __name__ == '__main__':

    input_shape = (3, 224, 224)

    config = configs[args.config]
    print(config)
    model = InternViT6B(in_chans=3,
                        patch_size=14,
                        img_size=224,
                        pretrain_size=224,
                        qkv_bias=False,
                        drop_path_rate=0.0,
                        embed_dim=config['embed_dim'],
                        num_heads=config['num_heads'],
                        mlp_ratio=config['mlp_ratio'],
                        init_values=0.1,
                        qk_normalization=True,
                        depth=config['depth'],
                        use_flash_attn=True,
                        with_cp=True,
                        freeze_vit=True,
                        cls_target='cls_patch_concat',
                        num_classes=0,
                        attn_pool_num_heads=16,
                        clip_embed_dim=768,
                        norm_type='rms').to(torch.bfloat16)

    for k, v in model.named_parameters():
        v.requires_grad = True

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    flops, params = get_flops(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

    image = torch.rand(128, 3, 224, 224).to(torch.bfloat16).cuda()
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(10)):
            out = model(image)
    torch.cuda.synchronize()
    end_time = time.time()

    print('warmup time: ', end_time - start_time)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for i in tqdm(range(50)):
            out = model(image)
    torch.cuda.synchronize()
    end_time = time.time()
    print('using time: ', (end_time - start_time))
    print('FPS: ', 50 * 128 / (end_time - start_time))
    print(config)
