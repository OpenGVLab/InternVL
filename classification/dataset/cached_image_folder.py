# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import io
import json
import logging
import math
import os
import os.path as osp
import re
import time
from abc import abstractmethod

import mmcv
import torch
import torch.distributed as dist
import torch.utils.data as data
from mmcv.fileio import FileClient
from PIL import Image
from tqdm import tqdm, trange

from .zipreader import ZipReader, is_zip_path

_logger = logging.getLogger(__name__)

_ERROR_RETRY = 50


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [
        d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
    ]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_dataset_with_ann(ann_file, img_prefix, extensions):
    images = []
    with open(ann_file, 'r') as f:
        contents = f.readlines()
        for line_str in contents:
            path_contents = [c for c in line_str.split('\t')]
            im_file_name = path_contents[0]
            class_index = int(path_contents[1])
            assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
            item = (os.path.join(img_prefix, im_file_name), class_index)
            images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

    root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self,
                 root,
                 loader,
                 extensions,
                 ann_file='',
                 img_prefix='',
                 transform=None,
                 target_transform=None,
                 cache_mode='no'):
        # image folder mode
        if ann_file == '':
            _, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions)
        # zip mode
        else:
            samples = make_dataset_with_ann(os.path.join(root, ann_file),
                                            os.path.join(root, img_prefix),
                                            extensions)

        if len(samples) == 0:
            raise (RuntimeError('Found 0 files in subfolders of: ' + root +
                                '\n' + 'Supported extensions are: ' +
                                ','.join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.labels = [y_1k for _, y_1k in samples]
        self.classes = list(set(self.labels))

        self.transform = transform
        self.target_transform = target_transform

        self.cache_mode = cache_mode
        if self.cache_mode != 'no':
            self.init_cache()

    def init_cache(self):
        assert self.cache_mode in ['part', 'full']
        n_sample = len(self.samples)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        samples_bytes = [None for _ in range(n_sample)]
        start_time = time.time()
        for index in range(n_sample):
            if index % (n_sample // 10) == 0:
                t = time.time() - start_time
                print(
                    f'global_rank {dist.get_rank()} cached {index}/{n_sample} takes {t:.2f}s per block'
                )
                start_time = time.time()
            path, target = self.samples[index]
            if self.cache_mode == 'full':
                samples_bytes[index] = (ZipReader.read(path), target)
            elif self.cache_mode == 'part' and index % world_size == global_rank:
                samples_bytes[index] = (ZipReader.read(path), target)
            else:
                samples_bytes[index] = (path, target)
        self.samples = samples_bytes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))

        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_img_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CachedImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

    root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root,
                 ann_file='',
                 img_prefix='',
                 transform=None,
                 target_transform=None,
                 loader=default_img_loader,
                 cache_mode='no'):
        super(CachedImageFolder,
              self).__init__(root,
                             loader,
                             IMG_EXTENSIONS,
                             ann_file=ann_file,
                             img_prefix=img_prefix,
                             transform=transform,
                             target_transform=target_transform,
                             cache_mode=cache_mode)
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImageCephDataset(data.Dataset):

    def __init__(self,
                 root,
                 split,
                 parser=None,
                 transform=None,
                 target_transform=None,
                 on_memory=False):
        if '22k' in root:
            # Imagenet 22k
            annotation_root = 'meta_data/'
        else:
            # Imagenet
            annotation_root = 'meta_data/'
        if parser is None or isinstance(parser, str):
            parser = ParserCephImage(root=root,
                                     split=split,
                                     annotation_root=annotation_root,
                                     on_memory=on_memory)
        self.parser = parser
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class Parser:

    def __init__(self):
        pass

    @abstractmethod
    def _filename(self, index, basename=False, absolute=False):
        pass

    def filename(self, index, basename=False, absolute=False):
        return self._filename(index, basename=basename, absolute=absolute)

    def filenames(self, basename=False, absolute=False):
        return [
            self._filename(index, basename=basename, absolute=absolute)
            for index in range(len(self))
        ]


class ParserCephImage(Parser):

    def __init__(self,
                 root,
                 split,
                 annotation_root,
                 on_memory=False,
                 **kwargs):
        super().__init__()

        self.file_client = None
        self.kwargs = kwargs

        self.root = root  # dataset:s3://imagenet22k
        if '22k' in root:
            self.io_backend = 'petrel'
            with open(osp.join(annotation_root, '22k_class_to_idx.json'),
                      'r') as f:
                self.class_to_idx = json.loads(f.read())
            with open(osp.join(annotation_root, '22k_label.txt'), 'r') as f:
                self.samples = f.read().splitlines()
        else:
            self.io_backend = 'disk'
            self.class_to_idx = None
            with open(osp.join(annotation_root, f'{split}.txt'), 'r') as f:
                self.samples = f.read().splitlines()
        local_rank = None
        local_size = None
        self._consecutive_errors = 0
        self.on_memory = on_memory
        if on_memory:
            self.holder = {}
            if local_rank is None:
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if local_size is None:
                local_size = int(os.environ.get('LOCAL_SIZE', 1))
            self.local_rank = local_rank
            self.local_size = local_size
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.num_replicas = int(os.environ['WORLD_SIZE'])
            self.num_parts = local_size
            self.num_samples = int(
                math.ceil(len(self.samples) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
            self.total_size_parts = self.num_samples * self.num_replicas // self.num_parts
            self.load_onto_memory_v2()

    def load_onto_memory(self):
        print('Loading images onto memory...', self.local_rank,
              self.local_size)
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        for index in trange(len(self.samples)):
            if index % self.local_size != self.local_rank:
                continue
            path, _ = self.samples[index].split(' ')
            path = osp.join(self.root, path)
            img_bytes = self.file_client.get(path)
            self.holder[path] = img_bytes

        print('Loading complete!')

    def load_onto_memory_v2(self):
        # print("Loading images onto memory...", self.local_rank, self.local_size)
        t = torch.Generator()
        t.manual_seed(0)
        indices = torch.randperm(len(self.samples), generator=t).tolist()
        # indices = range(len(self.samples))
        indices = [i for i in indices if i % self.num_parts == self.local_rank]
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size_parts - len(indices))]
        assert len(indices) == self.total_size_parts

        # subsample
        indices = indices[self.rank // self.num_parts:self.
                          total_size_parts:self.num_replicas // self.num_parts]
        assert len(indices) == self.num_samples

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        for index in tqdm(indices):
            if index % self.local_size != self.local_rank:
                continue
            path, _ = self.samples[index].split(' ')
            path = osp.join(self.root, path)
            img_bytes = self.file_client.get(path)

            self.holder[path] = img_bytes

        print('Loading complete!')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        filepath, target = self.samples[index].split(' ')
        filepath = osp.join(self.root, filepath)

        try:
            if self.on_memory:
                img_bytes = self.holder[filepath]
            else:
                # pass
                img_bytes = self.file_client.get(filepath)
            img = mmcv.imfrombytes(img_bytes)[:, :, ::-1]
        except Exception as e:
            _logger.warning(
                f'Skipped sample (index {index}, file {filepath}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self))
            else:
                raise e
        self._consecutive_errors = 0

        img = Image.fromarray(img)
        try:
            if self.class_to_idx is not None:
                target = self.class_to_idx[target]
            else:
                target = int(target)
        except:
            print(filepath, target)
            exit()

        return img, target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename, _ = self.samples[index].split(' ')
        filename = osp.join(self.root, filename)

        return filename


def get_temporal_info(date, miss_hour=False):
    try:
        if date:
            if miss_hour:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*)', re.I)
            else:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*) (\d*):(\d*):(\d*)',
                                     re.I)
            m = pattern.match(date.strip())

            if m:
                year = int(m.group(1))
                month = int(m.group(2))
                day = int(m.group(3))
                x_month = math.sin(2 * math.pi * month / 12)
                y_month = math.cos(2 * math.pi * month / 12)
                if miss_hour:
                    x_hour = 0
                    y_hour = 0
                else:
                    hour = int(m.group(4))
                    x_hour = math.sin(2 * math.pi * hour / 24)
                    y_hour = math.cos(2 * math.pi * hour / 24)
                return [x_month, y_month, x_hour, y_hour]
            else:
                return [0, 0, 0, 0]
        else:
            return [0, 0, 0, 0]
    except:
        return [0, 0, 0, 0]


def get_spatial_info(latitude, longitude):
    if latitude and longitude:
        latitude = math.radians(latitude)
        longitude = math.radians(longitude)
        x = math.cos(latitude) * math.cos(longitude)
        y = math.cos(latitude) * math.sin(longitude)
        z = math.sin(latitude)
        return [x, y, z]
    else:
        return [0, 0, 0]
