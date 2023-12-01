# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os

import torch
import torchvision.transforms as T
from torch import nn
from torchvision.transforms import InterpolationMode
from transformers import LlamaTokenizer

from .internvl_clip import InternVL_C

try:
    from .flash_attention import FlashAttention
except:
    print('flash attention is not installed.')


class InternVLTokenizer(nn.Module):
    def __init__(self, model_path):
        super(InternVLTokenizer, self).__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = ' '  # allow padding
        self.tokenizer.add_eos_token = True

    def forward(self, text, prefix='summarize:'):
        if type(text) == str:
            text = prefix + text
        elif type(text) == list:
            text = [prefix + item for item in text]
        text = self.tokenizer(text, return_tensors='pt', max_length=80, truncation=True, padding=True).input_ids
        return text


def process_checkpoint(ckpt):
    new_ckpt = {}
    for k, v in ckpt['module'].items():
        if 'bamboo' in k or 'predictor' in k or 'decoder' in k or 'loss' in k:
            continue
        new_k = k.replace('clip.transformer.', 'transformer.')
        new_k = new_k.replace('clip.text_projection', 'text_projection')
        new_k = new_k.replace('clip.logit_scale', 'logit_scale')

        new_ckpt[new_k] = v
    return new_ckpt


def build_transform(task, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if task == 'retrieval':
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])
    else:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])
    return transform


def get_model_and_transform(task, image_size, device):
    llama_path = os.path.split(os.path.realpath(__file__))[0]
    llama_path = os.path.join(llama_path, 'chinese_alpaca_lora_7b')
    model = InternVL_C(img_size=image_size, layerscale_force_fp32=True, llama_path=llama_path)
    model = model.to(device).to(torch.float16)
    transform = build_transform(task, image_size)
    return model, transform


def load_internvl_clip(model_name, pretrained, cache_dir, device, task):
    llm_path = '/mnt/petrelfs/share_data/chenzhe1/data/llm/chinese_alpaca_lora_7b'
    tokenizer = InternVLTokenizer(llm_path)
    model, transform = get_model_and_transform(task=task, image_size=224, device=device)
    ckpt_path = f'/mnt/petrelfs/wangwenhai/workspace_swj/code/sim_clip_wj_deepspeed/exp/configs/' \
                f'6b_vit/6b_vit_{model_name}_clip_alpaca_7b_laion5b_peak_1e-5_256gpu_all_trainable_degradation.sh/' \
                f'{pretrained}/mp_rank_00_model_states.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    new_ckpt = process_checkpoint(ckpt)
    message = model.load_state_dict(new_ckpt, strict=False)
    return model, transform, tokenizer
