# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import bisect
import copy
import logging
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from transformers.trainer_pt_utils import LabelSmoother

from .constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class PackedDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        data_rank,
        data_world_size,
        datasets: List,
        dataset_weight: List[int] = None,
        num_images_expected: int = 6,
        max_packed_tokens: int = 32768,
        max_buffer_size: int = 100,
        log_freq: int = 1000000,
        strict_mode: bool = False,
        debug_mode: bool = False,
        replacement: bool = True,
        allow_overflow: bool = True,
        allow_empty_data: bool = False,
        allow_deduplicated_ds_name: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.num_images_expected = num_images_expected
        self.max_buffer_size = max_buffer_size
        self.log_freq = log_freq
        self.strict_mode = strict_mode
        self.debug_mode = debug_mode
        self.replacement = replacement
        self.allow_overflow = allow_overflow
        self.allow_empty_data = allow_empty_data

        self.max_packed_tokens = max_packed_tokens

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        assert self.img_start_token_id != self.tokenizer.unk_token_id
        assert self.img_token_id != self.tokenizer.unk_token_id
        assert self.img_end_token_id != self.tokenizer.unk_token_id

        if dataset_weight is None:
            dataset_weight = [1] * len(datasets)
        self.dataset_type = [d.dataset_type for d in self.datasets]

        self.datasets_orig = datasets
        self.dataset_weight_orig = [w / sum(dataset_weight) for w in dataset_weight]

        self.datasets = [ds for ds in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]

        # lazy init
        self.worker_id = None
        self.worker_state_key = None
        self.dataset_iter_list = None
        self._state_dict = {
            'sample_info': {d.ds_name:0 for d in self.datasets},
        }

        self.worker_custom_infos = None

        ds_name_list = [d.ds_name for d in self.datasets]
        if not allow_deduplicated_ds_name:
            assert len(ds_name_list) == len(set(ds_name_list)), f'deduplicated ds_name: {ds_name_list}'

        for ds in self.datasets:
            if ds.max_num_images > self.num_images_expected:
                logger.warning(f'{ds.max_num_images=} of {ds.ds_name} is larger than {self.num_images_expected=}')
                ds.max_num_images = num_images_expected

            if ds.max_tokens > self.max_packed_tokens:
                logger.warning(f'{ds.max_tokens=} of {ds.ds_name} is larger than {self.max_packed_tokens=}')
                ds.max_tokens = self.max_packed_tokens

            self._state_dict[ds.ds_name] = {}

        if get_rank() == 0:
            logger.info(
                f'Loaded dataset to pack: {ds_name_list}, '
                f'{self.num_images_expected=}, {self.max_packed_tokens=}, '
                f'{self.replacement=}, {self.allow_overflow=}',
            )

            temp = []
            for ds, ds_w in zip(self.datasets, self.dataset_weight):
                temp.append(f'{ds.ds_name:<25}: {ds_w*100:.2f}%')
            temp = '\n'.join(temp)
            logger.info(
                f'Sampling prob for each dataset:\n{temp}'
            )

        if self.allow_empty_data:
            logger.warning('allow_empty_data is enabled, note that empty data may be generated!')

    def load_state_dict(self, state_dict, custom_infos=None):

        self.worker_custom_infos = custom_infos

        self._state_dict.update(state_dict)
        for ds in self.datasets:
            if ds.ds_name in self._state_dict:
                ds.load_state_dict(self._state_dict[ds.ds_name])
                logger.info(f'{ds.ds_name=} is resumed.')
            else:
                logger.warning(f'{ds.ds_name=} is not resumed.')

    def _should_log(self):
        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * get_rank() + worker_id
        num_workers = num_workers * get_world_size()

        return worker_id == 0

    def next_data(self, current_dataset_idx):
        while True:
            try:
                current_sample = next(self.dataset_iter_list[current_dataset_idx])
                break  # Exit loop if successful
            except StopIteration:
                if self.replacement:
                    # logger.info(f'[Worker id {self.worker_id}] Dataset {self.datasets[current_dataset_idx].ds_name} is exhausted, restart it.')
                    try:
                        self.dataset_iter_list[current_dataset_idx] = iter(self.datasets[current_dataset_idx])
                        current_sample = next(self.dataset_iter_list[current_dataset_idx])
                        break
                    except:
                        # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                        self.datasets.pop(current_dataset_idx)
                        self.dataset_iter_list.pop(current_dataset_idx)
                        self.dataset_weight.pop(current_dataset_idx)

                        if len(self.datasets) == 0:
                            raise StopIteration
                        current_dataset_idx = np.random.choice(len(self.datasets))
                else:
                    # logger.error(f'{self.worker_id=} Fail to get any data from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                    self.datasets.pop(current_dataset_idx)
                    self.dataset_iter_list.pop(current_dataset_idx)
                    self.dataset_weight.pop(current_dataset_idx)

                    if len(self.datasets) == 0:
                        raise StopIteration
                    current_dataset_idx = np.random.choice(len(self.datasets))
            except:
                logger.error('Unexpected error!')
                if len(self.datasets) == 0:
                    raise StopIteration
                current_dataset_idx = np.random.choice(len(self.datasets))

        current_ds_name = self.datasets[current_dataset_idx].ds_name
        current_sample['type_ids'] = torch.zeros_like(current_sample['input_ids']) + current_dataset_idx

        if self.worker_state_key not in self._state_dict[current_ds_name]:
            self._state_dict[current_ds_name][self.worker_state_key] = {}

        meta_info = current_sample.pop('meta_info', {})
        self._state_dict[current_ds_name][self.worker_state_key].update(**meta_info)
        self._state_dict['sample_info'][self.datasets[current_dataset_idx].ds_name] += 1
        return current_sample

    def find_buffer(self, buffer_list, new_sample):
        # NOTE: use `bisect` to search might be faster

        find = False
        find_idx = -1
        num_images_current = new_sample['pixel_values'].size(0)
        for buffer_idx, buffer in enumerate(buffer_list):
            num_images_buffer = buffer['pixel_values'].size(0)
            if num_images_buffer + num_images_current <= self.num_images_expected:
                num_merged_tokens = new_sample['input_ids'].size(0) + buffer['input_ids'].size(0)

                if num_merged_tokens <= self.max_packed_tokens:
                    find = True
                    find_idx = buffer_idx
                    break

                if self.allow_overflow and len(buffer_list) >= self.max_buffer_size // 2:
                    find = True
                    find_idx = buffer_idx

        if find:
            return buffer_list.pop(find_idx)
        return None

    def update_buffer(self, buffer, new_sample):
        if buffer is None:
            new_sample['data_index'] = torch.zeros_like(new_sample['input_ids'])
            return new_sample

        new_sample['data_index'] = torch.ones_like(new_sample['input_ids']) + buffer['data_index'][-1].item()

        assert buffer.keys() == new_sample.keys()
        for k in buffer:
            buffer[k] = torch.cat([buffer[k], new_sample[k]])
        return buffer

    @staticmethod
    def check_valid(sample_to_check, min_active_tokens_ratio=1/256):
        num_ignore_tokens = (sample_to_check['labels'] == IGNORE_TOKEN_ID).sum()
        num_tokens = sample_to_check['labels'].numel()
        return (1 - num_ignore_tokens / num_tokens) > min_active_tokens_ratio

    @staticmethod
    def split_buffer(buffer, max_tokens, img_start_token_id, img_token_id, img_end_token_id):
        if buffer['input_ids'].size(0) <= max_tokens:
            return [buffer]

        def _image_is_splitted(input_ids, cut_idx):
            is_image_start = input_ids[cut_idx].item() == img_start_token_id
            is_image_token = input_ids[cut_idx].item() == img_token_id
            is_image_end = input_ids[cut_idx].item() == img_end_token_id
            return is_image_start or is_image_token or is_image_end

        def _split(sample_to_split, left_idx, right_idx, left_img_idx, right_img_idx):
            assert (right_idx is None) == (right_img_idx is None)

            left_sample = {}
            right_sample = {} if right_idx is not None else None
            for k in sample_to_split:
                if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index', 'type_ids']:
                    left_sample[k] = sample_to_split[k][:left_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][right_idx:]
                elif k in ['pixel_values', 'image_flags']:
                    left_sample[k] = sample_to_split[k][:left_img_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][right_img_idx:]
                else:
                    raise NotImplementedError(f'find unsupported keys: {k} from {sample_to_split.keys()}')
            return left_sample, right_sample

        splitted_buffer = []
        while buffer['input_ids'].size(0) > max_tokens:
            img_start_idx_list = (buffer['input_ids'] == img_start_token_id).nonzero().squeeze(1).tolist()
            img_end_idx_list = (buffer['input_ids'] == img_end_token_id).nonzero().squeeze(1).tolist()
            assert len(img_start_idx_list) == len(img_end_idx_list)

            if _image_is_splitted(buffer['input_ids'], max_tokens):
                cut_idx = bisect.bisect_left(img_start_idx_list, max_tokens)
                if buffer['input_ids'][max_tokens] == img_start_token_id:
                    assert max_tokens == img_start_idx_list[cut_idx]
                    cut_left_idx = img_start_idx_list[cut_idx]
                    cut_left_img_idx = cut_idx
                else:
                    cut_left_idx = img_start_idx_list[cut_idx - 1]
                    cut_left_img_idx = cut_idx - 1
                cut_right_idx = cut_left_idx
                cut_right_img_idx = cut_left_img_idx
            else:
                cut_img_idx = bisect.bisect(img_start_idx_list, max_tokens)
                if cut_img_idx < len(img_start_idx_list):
                    cut_right_idx = img_start_idx_list[cut_img_idx]
                    cut_right_img_idx = cut_img_idx
                else:
                    cut_right_idx = None
                    cut_right_img_idx = None

                cut_left_idx = max_tokens
                cut_left_img_idx = cut_right_img_idx if cut_right_img_idx is not None else buffer['pixel_values'].size(0)

            left, right = _split(
                sample_to_split=buffer,
                left_idx=cut_left_idx,
                left_img_idx=cut_left_img_idx,
                right_idx=cut_right_idx,
                right_img_idx=cut_right_img_idx,
            )

            assert (left['input_ids'] == img_end_token_id).sum() == (left['input_ids'] == img_start_token_id).sum() == left['pixel_values'].size(0)
            if right is not None:
                assert (right['input_ids'] == img_end_token_id).sum() == (right['input_ids'] == img_start_token_id).sum() == right['pixel_values'].size(0)

            if left['pixel_values'].size(0) >= 1 and PackedDataset.check_valid(left):
                splitted_buffer.append(left)

            if right is None or right['pixel_values'].size(0) == 0:
                break

            buffer = right
            if buffer['input_ids'].size(0) <= max_tokens and PackedDataset.check_valid(buffer):
                splitted_buffer.append(buffer)
                break

        logger.debug(
            f'split a sample into {len(splitted_buffer)} samples, '
            f'current max_tokens={max_tokens}'
        )
        return splitted_buffer

    def update_buffer_list(self, buffer_list, buffer_max_len_list, buffer):
        # NOTE: in-place operation

        splitted_buffer = PackedDataset.split_buffer(
            buffer=buffer,
            max_tokens=self.max_packed_tokens,
            img_start_token_id=self.img_start_token_id,
            img_token_id=self.img_token_id,
            img_end_token_id=self.img_end_token_id,
        )

        for each_buffer in splitted_buffer:
            if each_buffer['pixel_values'].size(0) > self.num_images_expected:
                logger.error(
                    f"Find a sample with {each_buffer['pixel_values'].size(0)} images, "
                    f'which exceeds {self.num_images_expected}'
                )
                continue

            if each_buffer['input_ids'].size(0) >= self.max_packed_tokens:
                assert each_buffer['input_ids'].size(0) == self.max_packed_tokens
                buffer_max_len_list.append(each_buffer)
                continue

            find_idx = len(buffer_list)
            num_images_new_sample = each_buffer['pixel_values'].size(0)
            for buffer_idx in range(len(buffer_list)):
                if buffer_list[buffer_idx]['pixel_values'].size(0) < num_images_new_sample:
                    find_idx = buffer_idx
                    break
            buffer_list.insert(find_idx, each_buffer)

        for i in range(1, len(buffer_list)):
            assert buffer_list[i-1]['pixel_values'].size(0) >= buffer_list[i]['pixel_values'].size(0)

        return buffer_list, buffer_max_len_list

    def pad_buffer(self, buffer):
        if buffer['pixel_values'].size(0) == self.num_images_expected:
            return buffer

        num_pad_images = self.num_images_expected - buffer['pixel_values'].size(0)
        pad_images = torch.stack([
            torch.zeros_like(buffer['pixel_values'][0])
            for _ in range(num_pad_images)
        ])
        pad_image_flags = torch.tensor([0] * num_pad_images, dtype=torch.long)

        buffer['pixel_values'] = torch.cat([buffer['pixel_values'], pad_images])
        buffer['image_flags'] = torch.cat([buffer['image_flags'], pad_image_flags])

        return buffer

    def postprocess_buffer(self, buffer, custom_infos=None):
        buffer['worker_state_key'] = self.worker_state_key
        buffer['worker_state_dict'] = self._state_dict
        if custom_infos is not None:
            buffer['custom_infos'] = {self.worker_state_key: copy.deepcopy(custom_infos)}
        return buffer

    def print_log(self, iter_idx, buffer_list):
        if iter_idx % self.log_freq != 0:
            return

        if self._should_log():
            logger.info(
                f"{iter_idx=}, {len(buffer_list)=}, {self._state_dict['sample_info']}"
            )

    def __iter__(self):
        iter_idx = 0
        buffer_list = []
        buffer_max_len_list = []

        if self._should_log():
            logger.info(f'Begin to iter, {len(buffer_list)=}')

        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * self.data_rank + worker_id
        num_workers = num_workers * self.data_world_size

        rng = np.random.default_rng(seed=worker_id)

        # reset states of each dataset
        self.worker_id = worker_id
        self.worker_state_key = f'work_state_{self.worker_id}'
        self.datasets = [d for d in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]
        self.dataset_iter_list = [iter(d) for d in self.datasets]

        for ds in self.datasets:
            # if not isinstance(ds, (ImageTextPairDataset, InterleavedDataset)):
            ds.worker_id = worker_id
            ds.worker_state_key = f'work_state_{self.worker_id}'
            ds.num_workers = num_workers
            if self._should_log() and worker_id == 0:
                logger.info(f'set worker_id and num_workers of {ds.__class__.__name__} {ds.ds_name}')

        if self.worker_custom_infos is not None and self.worker_state_key in self.worker_custom_infos:
            custom_infos = self.worker_custom_infos[self.worker_state_key]
            # buffer list
            if 'buffer_list' in custom_infos and isinstance(custom_infos['buffer_list'], list):
                buffer_list = custom_infos['buffer_list']
                if self._should_log() and worker_id == 0:
                    logger.info(f'[{self.worker_state_key}] load buffer list --> {len(buffer_list)=}')
            # other infos

            # reset
            self.worker_custom_infos = None

        logger.debug(
            f'{self.__class__.__name__} Rank {self.data_rank} '
            f'Worker {worker_id} begin to load data'
        )

        while True:
            self.dataset_weight = [w / sum(self.dataset_weight) for w in self.dataset_weight]
            current_dataset_idx = rng.choice(len(self.dataset_iter_list), p=self.dataset_weight)

            try:
                current_sample = self.next_data(current_dataset_idx)
            except:
                logger.info(f'All datasets are exhausted, begin to empty the buffer_list ({len(buffer_list)=})')
                while len(buffer_list) > 0:
                    if self.strict_mode:
                        yield self.postprocess_buffer(self.pad_buffer(buffer_list.pop(0)))
                    else:
                        yield self.postprocess_buffer(buffer_list.pop(0))
                logger.info(f'buffer_list is empty! ({len(buffer_list)=})')
                return

            buffer = self.find_buffer(buffer_list, current_sample)
            buffer = self.update_buffer(buffer, current_sample)
            buffer_list, buffer_max_len_list = self.update_buffer_list(buffer_list, buffer_max_len_list, buffer)

            while len(buffer_max_len_list) > 0:
                if buffer_max_len_list[0]['pixel_values'].size(0) != self.max_packed_tokens:
                    logger.debug(
                        f'num tokens of a buffer exceed {self.max_packed_tokens=}, '
                        f"yield a sample with {buffer_max_len_list[0]['pixel_values'].size(0)} images"
                    )
                if self.strict_mode and buffer_max_len_list[0]['pixel_values'].size(0) != self.num_images_expected:
                    # buffer_max_len_list.pop(0)
                    yield self.postprocess_buffer(self.pad_buffer(buffer_max_len_list.pop(0)), {'buffer_list': buffer_list})
                else:
                    yield self.postprocess_buffer(buffer_max_len_list.pop(0), {'buffer_list': buffer_list})

            while len(buffer_list) > 0 and buffer_list[0]['pixel_values'].size(0) > self.num_images_expected:
                logger.error(
                    f"num images of a buffer ({buffer_list[0]['pixel_values'].size(0)}) "
                    f'is larger than num_images_expected({self.num_images_expected})'
                )
                buffer_list.pop(0)

            while len(buffer_list) > 0 and buffer_list[0]['pixel_values'].size(0) == self.num_images_expected:
                if self.debug_mode:
                    debug_data = self.postprocess_buffer(buffer_list.pop(0), {'buffer_list': buffer_list})
                    while True:
                        yield debug_data.copy()

                yield self.postprocess_buffer(buffer_list.pop(0), {'buffer_list': buffer_list})

            while len(buffer_list) > self.max_buffer_size:
                logger.debug(
                    f'Failed to pack data to exactly {self.num_images_expected} images, '
                    f"yield a data sample with {buffer_list[0]['pixel_values'].size(0)} images."
                )
                if self.strict_mode:
                    yield self.postprocess_buffer(self.pad_buffer(buffer_list.pop(0)), {'buffer_list': buffer_list})
                else:
                    yield self.postprocess_buffer(buffer_list.pop(0), {'buffer_list': buffer_list})

            self.print_log(iter_idx=iter_idx, buffer_list=buffer_list)
            iter_idx += 1

    @staticmethod
    def get_cu_seqlens_and_indexes(
        data_index: torch.LongTensor,  # (seq_len,)
        input_ids: torch.LongTensor,   # (seq_len,)
        labels: torch.LongTensor,   # (seq_len,)
        len2weight: callable,
    ):
        indexes = []
        cu_seqlens = [0]
        loss_weight = []

        start = data_index.min()
        end = data_index.max() + 1
        for i in range(start, end):
            num_tokens = (data_index == i).sum().item()
            indexes.extend(list(range(num_tokens)))
            cu_seqlens.append(cu_seqlens[-1] + num_tokens)
            assert num_tokens > 0

            curr_data_index = data_index[cu_seqlens[-2]:cu_seqlens[-2]+num_tokens]
            assert (curr_data_index == i).all(), data_index

            curr_labels = labels[cu_seqlens[-2]:cu_seqlens[-2]+num_tokens]
            num_effective_tokens = (curr_labels != IGNORE_TOKEN_ID).sum().item()
            loss_weight.extend([len2weight(num_effective_tokens)] * num_tokens)

        assert len(indexes) == data_index.size(0), f'{len(indexes)=}, {data_index.size(0)=}'

        loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
        return cu_seqlens, indexes, loss_weight


WARNING_CNT = defaultdict(int)


def packed_collate_fn(
    features,
    data_collator,
    len2weight: callable,
    max_item_length: int,
    micro_num: int = 1,
    loss_reduction_all_gather: bool = False,
    pad_id: int = 0,
):
    if not isinstance(features, list):
        features = [features]

    if len(features) > micro_num:
        raise NotImplementedError(f'{len(features)=} > {micro_num=}')

    if len(features) < micro_num and WARNING_CNT['micro_num_warning'] < 5:
        logger.warning(
            f'{len(features)=} > {micro_num=}, '
            f'the features will be padded to satisfy micro_num requirement'
        )
        WARNING_CNT['micro_num_warning'] += 1

    # ensure that the len(features) is equal to the required micro_num
    num_features = len(features)
    while len(features) < micro_num:
        features.append(copy.deepcopy(features[0]))
        features[-1]['labels'] = torch.full_like(features[-1]['labels'], IGNORE_TOKEN_ID)

    indexes = []
    cu_seqlens = []
    cu_num_images_list = [0]

    worker_state_key_list = []
    worker_state_dict_list = []
    worker_state_custom_infos_list = []

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]

    num_samples = 0
    num_padding_tokens = 0
    for feat_idx, feat in enumerate(features):
        data_index = feat.pop('data_index')
        curr_cu_seqlens, curr_indexes, curr_loss_weight = PackedDataset.get_cu_seqlens_and_indexes(
            data_index=data_index,
            input_ids=feat['input_ids'],
            labels=feat['labels'],
            len2weight=len2weight,
        )

        feat['loss_weight'] = curr_loss_weight

        if feat_idx < num_features:
            num_samples += len(curr_cu_seqlens) - 1

        if curr_cu_seqlens[-1] < max_item_length:
            curr_cu_seqlens.append(max_item_length)
            curr_indexes.extend(list(range(max_item_length - curr_cu_seqlens[-2])))

        indexes.append(torch.tensor(curr_indexes, dtype=torch.long))
        cu_seqlens.append(torch.tensor(curr_cu_seqlens, dtype=torch.int32))

        worker_state_key_list.append(feat.pop('worker_state_key'))
        worker_state_dict_list.append(feat.pop('worker_state_dict'))
        worker_state_custom_infos_list.append(feat.pop('custom_infos', None))

        num_padding_tokens += (max_item_length - feat['input_ids'].size(0))
        cu_num_images_list.append(cu_num_images_list[-1] + feat['pixel_values'].size(0))

    batch = data_collator(features=features, max_item_length=max_item_length, pad_id=pad_id)
    # convert it to list in case it is converted into bf16
    batch['loss_weight'] = torch.where(batch['labels'] == IGNORE_TOKEN_ID, 0, batch['loss_weight']).tolist()
    batch['attention_mask'] = torch.stack(cu_seqlens)
    batch['loss_reduction_all_gather'] = loss_reduction_all_gather
    batch['statistics'] = torch.tensor(
        [
            num_samples,
            num_padding_tokens,
            batch['image_flags'].numel() - batch['image_flags'].sum().item(),
        ],
        dtype=torch.long,
    )
    batch.pop('type_ids')
    return batch
