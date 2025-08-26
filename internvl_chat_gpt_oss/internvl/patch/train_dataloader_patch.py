# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import datasets
import torch
import transformers
from functools import partial
from typing import Optional, Callable
from torch.utils.data import DataLoader, Dataset
from transformers.trainer import is_datasets_available, seed_worker


def _get_dataloader(
    self,
    dataset: Dataset,
    description: str,
    batch_size: int,
    sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
    is_training: bool = False,
    dataloader_key: Optional[str] = None,
) -> DataLoader:
    """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

    data_collator = self.data_collator
    if is_datasets_available() and isinstance(dataset, datasets.Dataset):
        dataset = self._remove_unused_columns(dataset, description=description)
    else:
        data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(dataset, torch.utils.data.IterableDataset):
        if sampler_fn is not None:
            dataloader_params["sampler"] = sampler_fn(dataset)
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        if is_training:
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

    if self.args.split_annotations or getattr(self.args, 'use_packed_ds', False):
        print('split_annotations is enable, skip prepare dataloader')
        dataloader = DataLoader(dataset, **dataloader_params)
    else:
        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    # Store the prepared dataloader for subsequent evaluations if using persistent workers.
    if dataloader_key is not None and self.args.dataloader_persistent_workers:
        if hasattr(self, "_eval_dataloaders"):
            self._eval_dataloaders[dataloader_key] = dataloader
        else:
            self._eval_dataloaders = {dataloader_key: dataloader}

    return dataloader


def replace_train_dataloader():
    transformers.Trainer._get_dataloader = _get_dataloader
    print('Replace train dataloader!!')
