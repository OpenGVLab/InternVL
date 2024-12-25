# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MMVet v2`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### MMVet v2

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/mm-vet-v2 && cd data/mm-vet-v2

# Step 2: Download the dataset
wget https://github.com/yuweihao/MM-Vet/releases/download/v2/mm-vet-v2.zip
unzip mm-vet-v2.zip

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/mm-vet-v2
 ├── images
 └── mm-vet-v2.json
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
torchrun --nproc_per_node=8 eval/mmvetv2/evaluate_mmvet_v2.py --checkpoint ${CHECKPOINT} --dynamic
```

Alternatively, you can run the following simplified command:

```shell
GPUS=8 sh evaluate.sh ${CHECKPOINT} mmvetv2 --dynamic
```

After the test is completed, a file with a name similar to `results/mmvet-v2_241224214015.json` will be generated. Please upload this file to the [official server](https://huggingface.co/spaces/whyu/MM-Vet-v2_Evaluator) to obtain the evaluation scores.

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default      | Description                                                                                                       |
| ---------------- | ------ | ------------ | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`         | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'mmvet-v2'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`      | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`          | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`      | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`      | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
