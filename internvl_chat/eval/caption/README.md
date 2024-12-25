# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for image captioning across three datasets: `COCO`, `Flickr30k`, and `NoCaps`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### COCO Karpathy Test

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/coco && cd data/coco

# Step 2: Download and unzip image files
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip && unzip test2015.zip

# Step 3: Download and place the annotation files
mkdir -p annotations && cd annotations/
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test.json
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test_gt.json

cd ../../..
```

After preparation is complete, the directory structure is:

```shell
data/coco
├── annotations
│   ├── coco_karpathy_test.json
│   └── coco_karpathy_test_gt.json
├── train2014
├── val2014
└── test2015
```

### Flickr30K Karpathy Test

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/flickr30k && cd data/flickr30k

# Step 2: Download and unzip image files
# Download images from https://bryanplummer.com/Flickr30kEntities/

# Step 3: Download and place the annotation files
# Karpathy split annotations can be downloaded from the following link:
wget https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_test_karpathy.txt
# This file is provided by the clip-benchmark repository.
# We convert this txt file to json format, download the converted file:
wget https://github.com/OpenGVLab/InternVL/releases/download/data/flickr30k_test_karpathy.json

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/flickr30k
├── Images
├── flickr30k_test_karpathy.txt
└── flickr30k_test_karpathy.json
```

### NoCaps Val

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/nocaps && cd data/nocaps

# Step 2: Download and unzip image files
# Download images from https://nocaps.org/download

# Step 3: Download and place the annotation files
# Original annotations can be downloaded from https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/nocaps
├── images
└── nocaps_val_4500_captions.json
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
torchrun --nproc_per_node=8 eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --dynamic
```

Alternatively, you can run the following simplified command:

```shell
# Test COCO, Flickr30K, and NoCaps
GPUS=8 sh evaluate.sh ${CHECKPOINT} caption --dynamic
# Test COCO only
GPUS=8 sh evaluate.sh ${CHECKPOINT} caption-coco --dynamic
# Test Flickr30K only
GPUS=8 sh evaluate.sh ${CHECKPOINT} caption-flickr30k --dynamic
# Test NoCaps only
GPUS=8 sh evaluate.sh ${CHECKPOINT} caption-nocaps --dynamic
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default                   | Description                                                                                                       |
| ---------------- | ------ | ------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`                      | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'coco,flickr30k,nocaps'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`                   | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`                       | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`                   | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`                   | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
