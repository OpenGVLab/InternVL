# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .internvl_c import load_internvl_c

# from .internvl_c_hf import load_internvl_clip as load_internvl_clip_hf
# from .internvl_g_hf import load_internvl_qformer as load_internvl_qformer_hf


def load_internvl(model_name, pretrained, cache_dir, device):
    if model_name == 'internvl_c_classification':
        return load_internvl_c(pretrained, device, 'classification')
    elif model_name == 'internvl_c_retrieval':
        return load_internvl_c(pretrained, device, 'retrieval')
