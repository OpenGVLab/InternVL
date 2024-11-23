# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import datasets
import torch
import transformers
from torch.utils.data import DataLoader
from transformers.trainer import is_datasets_available, seed_worker


def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError('Trainer: training requires a train_dataset.')

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description='training')
    else:
        data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

    dataloader_params = {
        'batch_size': self._train_batch_size,
        'collate_fn': data_collator,
        'num_workers': self.args.dataloader_num_workers,
        'pin_memory': self.args.dataloader_pin_memory,
        'persistent_workers': self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params['sampler'] = self._get_train_sampler()
        dataloader_params['drop_last'] = self.args.dataloader_drop_last
        dataloader_params['worker_init_fn'] = seed_worker

    if self.args.use_packed_ds:
        return DataLoader(train_dataset, **dataloader_params)
    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


def replace_train_dataloader():
    transformers.Trainer.get_train_dataloader = get_train_dataloader
    # print('Replace train dataloader!!')
