# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from .ade import ADE20KDataset
from .pipelines import *  # noqa: F401,F403

__all__ = ['ADE20KDataset']
