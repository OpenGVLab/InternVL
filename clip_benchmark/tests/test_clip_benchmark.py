#!/usr/bin/env python

"""Tests for `clip_benchmark` package."""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from clip_benchmark.cli import run


class base_args:
    dataset = 'dummy'
    split = 'test'
    model = 'ViT-B-32-quickgelu'
    pretrained = 'laion400m_e32'
    task = 'zeroshot_classification'
    amp = False
    num_workers = 4
    batch_size = 64
    dataset_root = 'root'
    output = 'result.json'
    verbose = True
    root = 'root'
    annotation_file = ''
    seed = 0
    skip_load = False
    language = 'en'
    model_cache_dir = None
    cupl = False
    save_clf = None
    load_clfs = []
    model_type = 'open_clip'
    wds_cache_dir = None
    which = 'eval'
    skip_existing = False


def test_base():
    run(base_args)
