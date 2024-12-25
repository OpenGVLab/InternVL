# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MMVet`.

While the provided code can run the benchmark, we recommend using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for testing this benchmark if you aim to align results with our technical report.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### MMVet

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/mm-vet && cd data/mm-vet

# Step 2: Download the dataset
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/llava-mm-vet.jsonl
cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/mm-vet
 ├── images
 └── llava-mm-vet.jsonl
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 1-GPU setup:

```shell
torchrun --nproc_per_node=1 eval/mmvet/evaluate_mmvet.py --checkpoint ${CHECKPOINT} --dynamic
```

Alternatively, you can run the following simplified command:

```shell
GPUS=1 sh evaluate.sh ${CHECKPOINT} mmvet --dynamic
```

After the test is completed, a file with a name similar to `results/mmvet_241224214015.json` will be generated. Please upload this file to the [official server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator) to obtain the evaluation scores.

> ⚠️ Note: The test scores from the official server of MMVet will be significantly higher than those of VLMEvalKit. To align the scores with our technical report, please use VLMEvalKit to test this benchmark.

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default   | Description                                                                                                       |
| ---------------- | ------ | --------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`      | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'mmvet'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`   | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`       | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`   | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`   | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
