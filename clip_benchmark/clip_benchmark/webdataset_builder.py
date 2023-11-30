# Convert CLIP_benchmark datasets to webdataset format

import argparse
import io
import os
import sys

import torch
import torch.utils.data
import webdataset
from tqdm import tqdm

from .datasets.builder import build_dataset


def get_parser_args():
    parser = argparse.ArgumentParser(description="""
        Convert a CLIP_benchmark dataset to the webdataset format (TAR files).
        Datasets can be uploaded to the Huggingface Hub to allow CLIP model
        evaluation from anywhere with an Internet connection.

        To convert other image classification datasets, use the Python API:
        >>> import clip_benchmark.webdataset_builder
        >>> help(clip_benchmark.webdataset_builder.convert_dataset)
    """)
    # Main arguments
    parser.add_argument('--dataset', '-d', required=True, type=str,
                        help='CLIP_benchmark compatible dataset for conversion')
    parser.add_argument('--split', '-s', default='test', type=str,
                        help='Dataset split to use')
    parser.add_argument('--dataset-root', '-r', default='data', type=str,
                        help='Root directory for input data')
    parser.add_argument('--output', '-o', required=True, type=str,
                        help='Root directory for output data')
    # Special dataset types
    parser_special = parser.add_mutually_exclusive_group()
    parser_special.add_argument('--retrieval', action='store_true',
                                help='Flag to signal retrieval dataset (text captions instead of classes)')
    parser_special.add_argument('--multilabel', action='store_true',
                                help='Flag to signal multilabel classification dataset')
    # Additional parameters
    parser.add_argument('--image-format', default='webp', type=str,
                        help='Image extension for saving: (lossless) webp, png, or jpg (Default: webp)')
    parser.add_argument('--max-count', default=10_000, type=int,
                        help='Maximum number of images per TAR shard (Default: 10_000)')
    parser.add_argument('--max-size', default=1_000_000_000, type=int,
                        help='Maximum size in bytes per TAR shard (Default: 1_000_000_000)')
    args = parser.parse_args()
    return args


def main():
    args = get_parser_args()
    run(args)


def run(args):
    # Setup dataset folder
    os.makedirs(os.path.join(args.output, args.split), exist_ok=True)
    # Load original dataset
    dataset = build_dataset(
        dataset_name=args.dataset,
        root=args.dataset_root,
        split=args.split,
        transform=PIL_to_bytes(args.image_format),
        download=True,
    )
    # Run conversion
    if args.retrieval:
        convert_retrieval_dataset(
            dataset,
            args.split,
            args.output,
            transform=None,
            image_format=args.image_format,
            max_count=args.max_count,
            max_size=args.max_size
        )
    else:
        convert_dataset(
            dataset,
            args.split,
            args.output,
            transform=None,
            image_format=args.image_format,
            max_count=args.max_count,
            max_size=args.max_size,
            multilabel=args.multilabel,
        )


def PIL_to_bytes(image_format):
    OPTIONS = {
        'webp': dict(format='webp', lossless=True),
        'png': dict(format='png'),
        'jpg': dict(format='jpeg'),
    }

    def transform(image):
        bytestream = io.BytesIO()
        image.save(bytestream, **OPTIONS[image_format])
        return bytestream.getvalue()

    return transform


def path_to_bytes(filepath):
    with open(filepath, 'rb') as fp:
        return fp.read()


def convert_dataset(dataset, split, output_folder, *, transform=None,
                    image_format='webp', max_count=10_000, max_size=1_000_000_000,
                    multilabel=False, verbose=True):
    """
    Convert an iterable `dataset` of (image, label) pairs to webdataset (.tar) format, and store in `output_folder/split`.

    Images may be passed in as either:
    * File paths: pass in `transform=path_to_bytes`;
    * PIL images: pass in `transform=PIL_to_bytes(image_format)` where `image_format` is e.g. "webp"; or
    * Raw binary data: use a PyTorch `Dataset` that supports `transform=PIL_to_bytes(image_format)`, and pass in `transform=None` here.
        Be sure that the transform is not applied twice.

    Copying image files directly or writing raw binary data is fastest since it allows multiprocessing;
    passing in PIL images will be slower, but should work for any format of dataset.

    Labels must be zero-indexed integers (for multilabel datasets, labels must be arrays/tensors).

    Classnames and zero-shot classification templates can be provided as attributes of the dataset (`.classes` and `.templates`)
    or filled in manually afterward. `dataset.classes` should be a list of strings indexed by the labels,
    and `dataset.templates` should be a list of strings containing `{c}` to specify where classnames are to be inserted.
    """
    # Create output directory
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)
    # Multiprocessed dataloader, should work with Dataset or list
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=lambda batch: batch[0]  # No collate, only for multiprocessing
    )
    if verbose:
        try:
            print(f'Dataset size: {len(dataset)}')
        except TypeError:
            print('IterableDataset has no len()')
    # Save classnames
    if hasattr(dataset, 'classes') and dataset.classes:
        classnames_fname = os.path.join(output_folder, 'classnames.txt')
        with open(classnames_fname, 'w') as classnames_file:
            print(*dataset.classes, sep='\n', end='\n', file=classnames_file)
        if verbose:
            print("Saved class names to '%s'" % classnames_fname)
    elif verbose:
        print('WARNING: No class names found')
    # Save zeroshot templates
    if hasattr(dataset, 'templates') and dataset.templates:
        templates_fname = os.path.join(output_folder, 'zeroshot_classification_templates.txt')
        with open(templates_fname, 'w') as templates_file:
            print(*dataset.templates, sep='\n', end='\n', file=templates_file)
        if verbose:
            print("Saved class names to '%s'" % templates_fname)
    elif verbose:
        print('WARNING: No zeroshot classification templates found')
    # Save dataset type
    if multilabel:
        type_fname = os.path.join(output_folder, 'dataset_type.txt')
        with open(type_fname, 'w') as type_file:
            print('multilabel', end='\n', file=type_file)
            if verbose:
                print("Saved dataset type to '%s'" % type_fname)
    # Write to TAR files
    data_fname = os.path.join(output_folder, split, r'%d.tar')
    sink = webdataset.ShardWriter(
        data_fname,
        maxcount=max_count,
        maxsize=max_size
    )
    nsamples = 0
    label_type = 'npy' if multilabel else 'cls'
    for index, (input, output) in enumerate(tqdm(dataloader, desc='Converting')):
        nsamples += 1
        if isinstance(input, str) and transform is path_to_bytes:
            # If copying file, determine image format from extension
            extension = os.path.splitext(input)[1].replace('.', '').lower().replace('jpeg', 'jpg') or image_format
        else:
            extension = image_format
        # Convert label if necessary
        if isinstance(output, torch.Tensor):
            if multilabel:
                output = output.detach().cpu().numpy()
            else:
                output = output.item()
        # Write example
        sink.write({
            '__key__': 's%07d' % index,
            extension: transform(input) if transform else input,
            label_type: output,
        })
    num_shards = sink.shard
    sink.close()
    if verbose:
        print("Saved dataset to '%s'" % data_fname.replace(r'%d', '{0..%d}' % (num_shards - 1)))
    # Save number of shards
    nshards_fname = os.path.join(output_folder, split, 'nshards.txt')
    with open(nshards_fname, 'w') as nshards_file:
        print(num_shards, end='\n', file=nshards_file)
    if verbose:
        print("Saved number of shards = %d to '%s'" % (num_shards, nshards_fname))
    print('Final dataset size:', nsamples)


def convert_retrieval_dataset(dataset, split, output_folder, *, transform=None, image_format='webp', max_count=10_000,
                              max_size=1_000_000_000, verbose=True):
    """
    Convert an iterable `dataset` of (image, [caption1, caption2, ...]) pairs to webdataset (.tar) format, and store in `output_folder/split`.

    Labels must be lists of strings, with no newlines.

    Read the documentation of `convert_dataset` for more information.
    """
    # Create output directory
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)
    # Multiprocessed dataloader, should work with Dataset or list
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=lambda batch: batch[0]  # No collate, only for multiprocessing
    )
    if verbose:
        try:
            print(f'Dataset size: {len(dataset)}')
        except TypeError:
            print('IterableDataset has no len()')
    # No classnames
    # No zeroshot templates
    # Save dataset type
    type_fname = os.path.join(output_folder, 'dataset_type.txt')
    with open(type_fname, 'w') as type_file:
        print('retrieval', end='\n', file=type_file)
    if verbose:
        print("Saved dataset type to '%s'" % type_fname)
    # Write to TAR files
    data_fname = os.path.join(output_folder, split, r'%d.tar')
    sink = webdataset.ShardWriter(
        data_fname,
        maxcount=max_count,
        maxsize=max_size
    )
    nsamples = 0
    for index, (input, output) in enumerate(tqdm(dataloader, desc='Converting')):
        nsamples += 1
        if isinstance(input, str) and transform is path_to_bytes:
            # If copying file, determine image format from extension
            extension = os.path.splitext(input)[1].replace('.', '').lower().replace('jpeg', 'jpg') or image_format
        else:
            extension = image_format
        sink.write({
            '__key__': 's%07d' % index,
            extension: transform(input) if transform else input,
            'txt': '\n'.join(caption.replace('\n', r'\n') for caption in output),
        })
    num_shards = sink.shard
    sink.close()
    if verbose:
        print("Saved dataset to '%s'" % data_fname.replace(r'%d', '{0..%d}' % (num_shards - 1)))
    # Save number of shards
    nshards_fname = os.path.join(output_folder, split, 'nshards.txt')
    with open(nshards_fname, 'w') as nshards_file:
        print(num_shards, end='\n', file=nshards_file)
    if verbose:
        print("Saved number of shards = %d to '%s'" % (num_shards, nshards_fname))
    print('Final dataset size:', nsamples)


if __name__ == '__main__':
    sys.exit(main())
