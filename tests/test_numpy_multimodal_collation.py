import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def collators():
    path = Path(__file__).resolve().parents[1] / 'internvl_chat/internvl/patch/pad_data_collator.py'
    spec = importlib.util.spec_from_file_location('pad_data_collator_test_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature(length, images, offset, dpo=False, as_numpy=True):
    pixels = np.arange(images * 12, dtype=np.float32).reshape(images, 3, 2, 2) + offset
    flags = np.ones(images, dtype=np.int64)
    output = {'pixel_values': pixels, 'image_flags': flags}
    if not as_numpy:
        output = {key: torch.from_numpy(value) for key, value in output.items()}
    for prefix in ('chosen_', 'rejected_') if dpo else ('',):
        output[f'{prefix}input_ids'] = torch.arange(1, length + 1)
        output[f'{prefix}labels'] = torch.arange(1, length + 1)
    return output


@pytest.mark.parametrize('dpo', [False, True])
@pytest.mark.parametrize('image_counts', [(1, 3), (2, 2), (0, 2)])
def test_numpy_visual_arrays_concatenate_along_patch_axis(collators, dpo, image_counts):
    examples = [feature(length, count, offset, dpo) for length, count, offset in zip((3, 5), image_counts, (0, 100))]
    expected_pixels = np.concatenate([item['pixel_values'] for item in examples], axis=0)
    expected_flags = np.concatenate([item['image_flags'] for item in examples], axis=0)
    collate = collators.dpo_concat_pad_data_collator if dpo else collators.concat_pad_data_collator
    result = collate(examples)
    torch.testing.assert_close(result['pixel_values'], torch.from_numpy(expected_pixels))
    torch.testing.assert_close(result['image_flags'], torch.from_numpy(expected_flags))
    assert result['pixel_values'].dtype == torch.float32
    assert result['image_flags'].dtype == torch.int64
    prefix = 'chosen_' if dpo else ''
    assert result[f'{prefix}input_ids'].shape == (2, 5)
    assert result[f'{prefix}labels'][0, 3:].tolist() == [-100, -100]


@pytest.mark.parametrize('dpo', [False, True])
def test_tensor_visual_inputs_keep_their_existing_behavior(collators, dpo):
    examples = [feature(3, 1, 0, dpo, False), feature(4, 2, 100, dpo, False)]
    expected = torch.cat([item['pixel_values'] for item in examples])
    collate = collators.dpo_concat_pad_data_collator if dpo else collators.concat_pad_data_collator
    torch.testing.assert_close(collate(examples)['pixel_values'], expected)
