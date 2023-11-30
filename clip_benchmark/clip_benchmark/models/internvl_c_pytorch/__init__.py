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

from .internvl_c import InternVL_C

try:
    from .flash_attention import FlashAttention
except:
    print('FlashAttention is not installed.')


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
    llm_path = os.path.split(os.path.realpath(__file__))[0]
    llm_path = os.path.join(llm_path, 'chinese_alpaca_lora_7b')
    model = InternVL_C(img_size=image_size, layerscale_force_fp32=True, llm_path=llm_path)
    model = model.to(torch.float16).to(device)
    transform = build_transform(task, image_size)
    return model, transform


def load_internvl_c_pytorch(ckpt_path, device, task, image_size=224):
    llm_path = os.path.split(os.path.realpath(__file__))[0]
    llm_path = os.path.join(llm_path, 'chinese_alpaca_lora_7b')
    tokenizer = InternVLTokenizer(llm_path)
    model, transform = get_model_and_transform(task=task, image_size=image_size, device=device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    return model, transform, tokenizer
