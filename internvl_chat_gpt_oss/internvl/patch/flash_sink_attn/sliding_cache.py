# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# This file contains code originally written by Wenhao Li.
# --------------------------------------------------------

import torch


class SlidingCacheManager:
    """
    专门实现这个类, 因为推理的时候要取local window
    """
    def __init__(
            self, 
            sliding_window: int = 256):
        

        super().__init__()
        self.sliding_window = sliding_window
        self.reset()

    def reset(self):
        self.num_kv = 0
        self.key = None
        self.val = None

    def update(self, key, val):
        self.key = key
        self.val = val
        self.num_kv = key.shape[1]
