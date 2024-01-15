# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from transformers import LlamaTokenizer

from .configuration_intern_vit import InternVisionConfig
from .configuration_internvl import InternVLConfig
from .modeling_intern_vit import InternVisionModel
from .modeling_internvl import InternVL_C, InternVL_G, InternVLModel

__all__ = ['InternVisionConfig', 'InternVisionModel', 'InternVLConfig',
           'InternVLModel', 'InternVL_C', 'InternVL_G']


# Prefix the text "summarize:"
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
        text = self.tokenizer(text, return_tensors='pt', max_length=80, truncation=True, padding='max_length').input_ids
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
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform


def load_internvl_c_huggingface(ckpt_path, device, task):
    model = InternVL_C.from_pretrained(ckpt_path, torch_dtype=torch.float16).to(device)
    if model.config.use_backbone_lora:
        model.vision_model.merge_and_unload()
        model.vision_model = model.vision_model.model
    if model.config.use_qllama_lora:
        model.qllama.merge_and_unload()
        model.qllama = model.qllama.model
    if model.config.force_image_size is not None:
        image_size = model.config.force_image_size
    else:
        image_size = model.config.vision_config.image_size
    transform = build_transform(task, image_size)
    tokenizer = InternVLTokenizer(ckpt_path)
    return model, transform, tokenizer


def load_internvl_g_huggingface(ckpt_path, device, task):
    model = InternVL_G.from_pretrained(ckpt_path, torch_dtype=torch.float16).to(device)
    if model.config.use_backbone_lora:
        model.vision_model.merge_and_unload()
        model.vision_model = model.vision_model.model
    if model.config.use_qllama_lora:
        model.qllama.merge_and_unload()
        model.qllama = model.qllama.model
    if model.config.force_image_size is not None:
        image_size = model.config.force_image_size
    else:
        image_size = model.config.vision_config.image_size
    transform = build_transform(task, image_size)
    tokenizer = InternVLTokenizer(ckpt_path)
    return model, transform, tokenizer
