# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .internvl_c_pytorch import load_internvl_c_pytorch
from .internvl_huggingface import (load_internvl_c_huggingface,
                                   load_internvl_g_huggingface)


def load_internvl(model_name, pretrained, cache_dir, device):
    if model_name == 'internvl_c_classification':
        return load_internvl_c_pytorch(pretrained, device, 'classification')
    elif model_name == 'internvl_c_retrieval':
        return load_internvl_c_pytorch(pretrained, device, 'retrieval')
    elif model_name == 'internvl_c_classification_hf':
        return load_internvl_c_huggingface(pretrained, device, 'classification')
    elif model_name == 'internvl_c_retrieval_hf':
        return load_internvl_c_huggingface(pretrained, device, 'retrieval')
    elif model_name == 'internvl_g_classification_hf':
        return load_internvl_g_huggingface(pretrained, device, 'classification')
    elif model_name == 'internvl_g_retrieval_hf':
        return load_internvl_g_huggingface(pretrained, device, 'retrieval')
