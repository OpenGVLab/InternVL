import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from transformers import LlamaTokenizer

from .configuration_intern_clip import InternCLIPConfig, InternVisionConfig
from .modeling_intern_clip import (InternCLIPModel,
                                   InternCLIPModelForBenchmark,
                                   InternVisionModel)

__all__ = ['InternVisionConfig', 'InternCLIPConfig', 'InternCLIPModel',
           'InternVisionModel', 'InternCLIPModelForBenchmark']


# Prefix the text "summarize:"
class InternCLIPTokenizer(nn.Module):
    def __init__(self, model_path):
        super(InternCLIPTokenizer, self).__init__()
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
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform


def load_internvl_clip(model_name, pretrained, cache_dir, device, task):
    model_path = '/mnt/petrelfs/share_data/wangwenhai/internvl/internvl_huggingface/intern_clip_13b'
    model = InternCLIPModelForBenchmark.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    transform = build_transform(task)
    tokenizer = InternCLIPTokenizer(model_path)
    return model, transform, tokenizer
