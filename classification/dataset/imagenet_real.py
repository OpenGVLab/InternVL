# --------------------------------------------------------
# EVA: Exploring the Limits of Masked Visual Representation Learning at Scale (https://arxiv.org/abs/2211.07636)
# Github source: https://github.com/baaivision/EVA
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
# Based on timm, DINO, DeiT and BEiT codebases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------'

import json
import os

import numpy as np


class RealLabelsImagenet:

    def __init__(self, filenames, real_json='real.json', topk=(1, 5)):
        with open(real_json) as real_labels:
            real_labels = json.load(real_labels)
            real_labels = {f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels for i, labels in enumerate(real_labels)}
        self.real_labels = real_labels
        self.filenames = filenames
        assert len(self.filenames) == len(self.real_labels)
        self.topk = topk
        self.is_correct = {k: [] for k in topk}
        self.sample_idx = 0

    def add_result(self, output):
        maxk = max(self.topk)
        _, pred_batch = output.topk(maxk, 1, True, True)
        pred_batch = pred_batch.cpu().numpy()
        for pred in pred_batch:
            filename = self.filenames[self.sample_idx]
            filename = os.path.basename(filename)
            if self.real_labels[filename]:
                for k in self.topk:
                    self.is_correct[k].append(
                        any([p in self.real_labels[filename] for p in pred[:k]]))
            self.sample_idx += 1

    def get_accuracy(self, k=None):
        if k is None:
            return {k: float(np.mean(self.is_correct[k] for k in self.topk))}
        else:
            return float(np.mean(self.is_correct[k])) * 100
