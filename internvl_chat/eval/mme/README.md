# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MME`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### MME

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/mme && cd data/mme

# Step 2: Download MME_Benchmark_release_version
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/MME_Benchmark_release_version.zip
unzip MME_Benchmark_release_version.zip

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/mme
 └── MME_Benchmark_release_version
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 1-GPU setup:

```shell
cd eval/mme/
DIRNAME=`basename ${CHECKPOINT}`
python eval.py --checkpoint ${CHECKPOINT} --dynamic
python calculation.py --results_dir ${DIRNAME}
cd ../../
```

Alternatively, you can run the following simplified command:

```shell
GPUS=1 sh evaluate.sh ${CHECKPOINT} mme --dynamic
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default | Description                                                                                                       |
| ---------------- | ------ | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`    | Path to the model checkpoint.                                                                                     |
| `--dynamic`      | `flag` | `False` | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`     | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False` | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False` | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
