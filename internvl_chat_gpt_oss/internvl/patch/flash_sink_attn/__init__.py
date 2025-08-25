# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# This file contains code originally written by Wenhao Li.
# --------------------------------------------------------

from .flash_sink_attn_gpt_oss import flash_sink_attn_func
from .flash_sink_varlen_attn_gpt_oss import flash_sink_attn_varlen_func
from .sliding_cache import SlidingCacheManager