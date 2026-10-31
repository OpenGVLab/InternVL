import importlib
import random
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def dataset(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'internvl_chat'))
    return importlib.import_module('internvl.train.dataset')


def test_random_sampling_can_select_each_interval_endpoint(dataset, monkeypatch):
    intervals = []

    def choose_last(candidates):
        candidates = list(candidates)
        intervals.append(candidates)
        return candidates[-1]

    monkeypatch.setattr(dataset.random, 'choice', choose_last)
    assert dataset.get_frame_indices(3, 10, sample='rand') == [2, 5, 9]
    assert intervals == [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]


def test_singleton_intervals_keep_stratification(dataset, monkeypatch):
    def unexpected_permutation(*args):
        pytest.fail('Singleton intervals must not trigger global permutation')

    monkeypatch.setattr(dataset.np.random, 'permutation', unexpected_permutation)
    monkeypatch.setattr(dataset.random, 'choice', lambda candidates: candidates[-1])
    assert dataset.get_frame_indices(4, 6, sample='rand') == [0, 2, 3, 5]


def test_short_video_pads_with_its_last_frame(dataset, monkeypatch):
    monkeypatch.setattr(dataset.random, 'choice', lambda candidates: candidates[-1])
    assert dataset.get_frame_indices(5, 2, sample='rand') == [0, 1, 1, 1, 1]


@pytest.mark.parametrize('seed', range(5))
def test_random_samples_stay_ordered_with_one_from_each_bin(dataset, seed):
    state = random.getstate()
    try:
        random.seed(seed)
        indices = dataset.get_frame_indices(7, 31, sample='rand')
    finally:
        random.setstate(state)
    bounds = np.linspace(0, 31, 8).astype(int)
    assert len(indices) == 7
    assert indices == sorted(indices)
    assert all(start <= index < stop for index, start, stop in zip(indices, bounds[:-1], bounds[1:]))


def test_middle_sampling_is_unchanged(dataset):
    assert dataset.get_frame_indices(3, 10, sample='middle') == [1, 4, 7]
