from functools import partial
from typing import Union

import torch

from .internvl_c import load_internvl_clip as load_internvl_clip
from .internvl_c_hf import load_internvl_clip as load_internvl_clip_hf
from .internvl_g_hf import load_internvl_qformer as load_internvl_qformer_hf
from .japanese_clip import load_japanese_clip
from .open_clip import load_open_clip

# loading function must return (model, transform, tokenizer)
TYPE2FUNC = {
    'open_clip': load_open_clip,
    'ja_clip': load_japanese_clip,
    'internvl_clip_retrieval': partial(load_internvl_clip, task='retrieval'),
    'internvl_clip_classification': partial(load_internvl_clip, task='classification'),
    'internvl_clip_hf_retrieval': partial(load_internvl_clip_hf, task='retrieval'),
    'internvl_clip_hf_classification': partial(load_internvl_clip_hf, task='classification'),
    'internvl_qformer_hf_retrieval': partial(load_internvl_qformer_hf, task='retrieval'),
    'internvl_qformer_hf_classification': partial(load_internvl_qformer_hf, task='classification'),
}
MODEL_TYPES = list(TYPE2FUNC.keys())


def load_clip(
        model_type: str,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        device: Union[str, torch.device] = 'cuda'
):
    assert model_type in MODEL_TYPES, f'model_type={model_type} is invalid!'
    load_func = TYPE2FUNC[model_type]
    return load_func(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device)
