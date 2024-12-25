# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MP-DocVQA`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### MP-DocVQA

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/mpdocvqa && cd data/mpdocvqa

# Step 2: Download the dataset
# Download from https://rrc.cvc.uab.es/?ch=17&com=downloads

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/mpdocvqa
 ├── images
 ├── test.json
 ├── train.json
 └── val.json
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
# Test the val set
torchrun --nproc_per_node=8 eval/mpdocvqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --dynamic --max-num 18
# Test the test set
torchrun --nproc_per_node=8 eval/mpdocvqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets mpdocvqa_val --dynamic --max-num 18
```

Alternatively, you can run the following simplified command:

```shell
# Test the val set
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-mpdocvqa-val --dynamic --max-num 18
# Test the test set
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-mpdocvqa-test --dynamic --max-num 18
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default          | Description                                                                                                       |
| ---------------- | ------ | ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`             | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'mpdocvqa_val'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`          | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `18`             | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`          | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`          | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
