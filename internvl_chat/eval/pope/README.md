# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `POPE`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### POPE

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/pope && cd data/pope

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/val2014 ./
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_pope_test.jsonl

# Step 3: Download `coco` from POPE
mkdir -p coco && cd coco
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json
cd ../../..
```

After preparation is complete, the directory structure is:

```shell
data/pope
├── coco
│   ├── coco_pope_adversarial.json
│   ├── coco_pope_popular.json
│   └── coco_pope_random.json
├── llava_pope_test.jsonl
└── val2014
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
torchrun --nproc_per_node=8 eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --dynamic
```

Alternatively, you can run the following simplified command:

```shell
GPUS=8 sh evaluate.sh ${CHECKPOINT} pope --dynamic
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default  | Description                                                                                                       |
| ---------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`     | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'pope'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`  | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`      | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`  | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`  | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
