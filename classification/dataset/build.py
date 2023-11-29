# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os

import numpy as np
import torch
import torch.distributed as dist
from timm.data import Mixup, create_transform
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .cached_image_folder import ImageCephDataset
from .samplers import NodeDistributedSampler, SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


class TTA(torch.nn.Module):

    def __init__(self, size, scales=[1.0, 1.05, 1.1]):
        super().__init__()
        self.size = size
        self.scales = scales

    def forward(self, img):
        out = []
        cc = transforms.CenterCrop(self.size)
        for scale in self.scales:
            size_ = int(scale * self.size)
            rs = transforms.Resize(size_, interpolation=_pil_interp('bicubic'))
            img_ = rs(img)
            img_ = cc(img_)
            out.append(img_)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, scale={self.scales})'


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset('train', config=config)
    config.freeze()
    print(f'local rank {config.LOCAL_RANK} / global rank {dist.get_rank()}'
          'successfully build train dataset')

    dataset_val, _ = build_dataset('val', config=config)
    print(f'local rank {config.LOCAL_RANK} / global rank {dist.get_rank()}'
          'successfully build val dataset')

    dataset_test, _ = build_dataset('test', config=config)
    print(f'local rank {config.LOCAL_RANK} / global rank {dist.get_rank()}'
          'successfully build test dataset')

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    if dataset_train is not None:
        if config.DATA.IMG_ON_MEMORY:
            sampler_train = NodeDistributedSampler(dataset_train)
        else:
            if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
                indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
                sampler_train = SubsetRandomSampler(indices)
            else:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train,
                    num_replicas=num_tasks,
                    rank=global_rank,
                    shuffle=True)

    if dataset_val is not None:
        if config.TEST.SEQUENTIAL:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)

    if dataset_test is not None:
        if config.TEST.SEQUENTIAL:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True) if dataset_train is not None else None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_val is not None else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_test is not None else None

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=config.AUG.MIXUP,
                         cutmix_alpha=config.AUG.CUTMIX,
                         cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                         prob=config.AUG.MIXUP_PROB,
                         switch_prob=config.AUG.MIXUP_SWITCH_PROB,
                         mode=config.AUG.MIXUP_MODE,
                         label_smoothing=config.MODEL.LABEL_SMOOTHING,
                         num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, dataset_test, data_loader_train, \
           data_loader_val, data_loader_test, mixup_fn


def build_loader2(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset('train', config=config)
    config.freeze()
    dataset_val, _ = build_dataset('val', config=config)
    dataset_test, _ = build_dataset('test', config=config)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True) if dataset_train is not None else None

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_val is not None else None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=True) if dataset_test is not None else None

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=config.AUG.MIXUP,
                         cutmix_alpha=config.AUG.CUTMIX,
                         cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                         prob=config.AUG.MIXUP_PROB,
                         switch_prob=config.AUG.MIXUP_SWITCH_PROB,
                         mode=config.AUG.MIXUP_MODE,
                         label_smoothing=config.MODEL.LABEL_SMOOTHING,
                         num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, dataset_test, data_loader_train, \
           data_loader_val, data_loader_test, mixup_fn


def build_dataset(split, config):
    if config.DATA.TRANSFORM == 'build_transform':
        transform = build_transform(split == 'train', config)
    elif config.DATA.TRANSFORM == 'build_transform_for_linear_probe':
        transform = build_transform_for_linear_probe(split == 'train', config)
    else:
        raise NotImplementedError
    print(split, transform)
    dataset = None
    nb_classes = None
    prefix = split
    if config.DATA.DATASET == 'imagenet' or config.DATA.DATASET == 'imagenet-real':
        if prefix == 'train' and not config.EVAL_MODE:
            root = os.path.join(config.DATA.DATA_PATH, 'train')
            dataset = ImageCephDataset(root, 'train',
                                       transform=transform,
                                       on_memory=config.DATA.IMG_ON_MEMORY)
        elif prefix == 'val':
            root = os.path.join(config.DATA.DATA_PATH, 'val')
            dataset = ImageCephDataset(root, 'val', transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        if prefix == 'train':
            if not config.EVAL_MODE:
                root = config.DATA.DATA_PATH
                dataset = ImageCephDataset(root, 'train',
                                           transform=transform,
                                           on_memory=config.DATA.IMG_ON_MEMORY)
            nb_classes = 21841
        elif prefix == 'val':
            root = os.path.join(config.DATA.DATA_PATH, 'val')
            dataset = ImageCephDataset(root, 'val', transform=transform)
            nb_classes = 1000
    elif config.DATA.DATASET == 'imagenetv2':
        from .imagenetv2 import ImageNetV2Dataset
        if prefix == 'train' and not config.EVAL_MODE:
            print(f'Only test split available for {config.DATA.DATASET}')
        else:
            dataset = ImageNetV2Dataset(variant='matched-frequency',
                                        transform=transform,
                                        location=config.DATA.DATA_PATH)
            nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet_sketch':
        if prefix == 'train' and not config.EVAL_MODE:
            print(f'Only test split available for {config.DATA.DATASET}')
        else:
            dataset = ImageFolder(root=config.DATA.DATA_PATH, transform=transform)
            nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet_a':
        if prefix == 'train' and not config.EVAL_MODE:
            print(f'Only test split available for {config.DATA.DATASET}')
        else:
            dataset = ImageFolder(root=config.DATA.DATA_PATH, transform=transform)
            nb_classes = 1000  # actual number of classes is 200
    elif config.DATA.DATASET == 'imagenet_r':
        if prefix == 'train' and not config.EVAL_MODE:
            print(f'Only test split available for {config.DATA.DATASET}')
        else:
            dataset = ImageFolder(root=config.DATA.DATA_PATH, transform=transform)
            nb_classes = 1000  # actual number of classes is 200
    else:
        raise NotImplementedError(
            f'build_dataset does support {config.DATA.DATASET}')

    return dataset, nb_classes


def build_transform_for_linear_probe(is_train, config):
    # linear probe: weak augmentation
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                config.DATA.IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.AUG.MEAN, std=config.AUG.STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(
                config.DATA.IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.DATA.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.AUG.MEAN, std=config.AUG.STD)
        ])
    return transform


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER
            if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT
            if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)

        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int(1.0 * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        elif config.AUG.RANDOM_RESIZED_CROP:
            t.append(
                transforms.RandomResizedCrop(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION)))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=_pil_interp(config.DATA.INTERPOLATION)))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(config.AUG.MEAN, config.AUG.STD))

    return transforms.Compose(t)
