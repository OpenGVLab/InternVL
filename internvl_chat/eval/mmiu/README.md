# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MMIU`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### MMIU

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/mmiu && cd data/mmiu

# Step 2: Download images
wget https://huggingface.co/MMIUBenchmark/MMIU/resolve/main/2D-spatial.zip
wget https://huggingface.co/MMIUBenchmark/MMIU/resolve/main/3D-spatial.zip
unzip 2D-spatial.zip
unzip 3D-spatial.zip

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/mmiu
 ├── 2D-spatial
 └── 3D-spatial
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
torchrun --nproc_per_node=8 eval/mmiu/evaluate_mmiu.py --checkpoint ${CHECKPOINT} --dynamic --max-num 12
```

Alternatively, you can run the following simplified command:

```shell
GPUS=8 sh evaluate.sh ${CHECKPOINT} mmiu --dynamic --max-num 12
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default  | Description                                                                                                       |
| ---------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`     | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'mmiu'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`  | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `12`     | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`  | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`  | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
