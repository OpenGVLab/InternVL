import importlib
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def dataset(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'internvl_chat'))
    return importlib.import_module('internvl.train.dataset')


def make_frames(folder, count):
    for index in reversed(range(count)):
        Image.new('RGB', (8, 6), (index, 0, 0)).save(folder / f'frame_{index}.jpg', format='PNG')


def test_local_folder_decodes_only_selected_frames(dataset, tmp_path, monkeypatch):
    make_frames(tmp_path, 64)
    original_open = Image.open
    opened = []

    def track_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(dataset.Image, 'open', track_open)
    frames = dataset.read_frames_folder(str(tmp_path), 4, min_num_frames=4, sample='middle')
    assert opened == ['frame_7.jpg', 'frame_23.jpg', 'frame_39.jpg', 'frame_55.jpg']
    assert [image.getpixel((0, 0))[0] for image in frames] == [7, 23, 39, 55]
    assert all(image.mode == 'RGB' for image in frames)


def test_unselected_corrupt_frame_does_not_fail_loading(dataset, tmp_path):
    make_frames(tmp_path, 8)
    (tmp_path / 'frame_0.jpg').write_bytes(b'corrupt unselected frame')
    frames = dataset.read_frames_folder(str(tmp_path), 2, min_num_frames=2, sample='middle')
    assert [image.getpixel((0, 0))[0] for image in frames] == [1, 5]


def test_remote_folder_fetches_only_selected_frames_and_converts_rgb(dataset):
    class Client:
        def __init__(self):
            self.requested = []

        def list(self, path):
            return [f'frame_{index}.jpg' for index in reversed(range(12))]

        def get(self, path):
            self.requested.append(path)
            index = dataset.extract_frame_number(path)
            output = BytesIO()
            Image.new('L', (8, 6), index).save(output, format='PNG')
            return output.getvalue()

    client = Client()
    frames = dataset.read_frames_folder('s3://bucket/frames', 3, min_num_frames=3, sample='middle', client=client)
    assert client.requested == [f's3://bucket/frames/frame_{index}.jpg' for index in [1, 5, 9]]
    assert [image.getpixel((0, 0)) for image in frames] == [(1, 1, 1), (5, 5, 5), (9, 9, 9)]


def test_short_folder_keeps_all_frames_in_order(dataset, tmp_path):
    make_frames(tmp_path, 2)
    frames = dataset.read_frames_folder(str(tmp_path), 4, min_num_frames=4, sample='middle')
    assert [image.getpixel((0, 0))[0] for image in frames] == [0, 1]


def test_empty_folder_still_returns_no_frames(dataset, tmp_path):
    assert dataset.read_frames_folder(str(tmp_path), 4, min_num_frames=4, sample='middle') == []
