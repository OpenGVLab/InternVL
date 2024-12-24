# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MMBench` and `CCBench`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### MMBench and CCBench

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/mmbench && cd data/mmbench

# Step 2: Download csv files
wget http://opencompass.openxlab.space/utils/MMBench/CCBench_legacy.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_en_20231003.tsv

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/mmbench
 ├── CCBench_legacy.tsv
 ├── mmbench_dev_20230712.tsv
 ├── mmbench_dev_cn_20231003.tsv
 ├── mmbench_dev_en_20231003.tsv
 ├── mmbench_test_cn_20231003.tsv
 └── mmbench_test_en_20231003.tsv
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
# Test the MMBench-Dev-EN
torchrun --nproc_per_node=8 eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --dynamic --datasets mmbench_dev_20230712
# Test the MMBench-Test-EN
torchrun --nproc_per_node=8 eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --dynamic --datasets mmbench_test_en_20231003
# Test the MMBench-Dev-CN
torchrun --nproc_per_node=8 eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --dynamic --datasets mmbench_dev_cn_20231003
# Test the MMBench-Test-CN
torchrun --nproc_per_node=8 eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --dynamic --datasets mmbench_test_cn_20231003
# Test the CCBench-Dev
torchrun --nproc_per_node=8 eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --dynamic --datasets ccbench_dev_cn
```

Alternatively, you can run the following simplified command:

```shell
# Test the MMBench-Dev-EN
GPUS=8 sh evaluate.sh ${CHECKPOINT} mmbench-dev-en --dynamic
# Test the MMBench-Test-EN
GPUS=8 sh evaluate.sh ${CHECKPOINT} mmbench-test-en --dynamic
# Test the MMBench-Dev-CN
GPUS=8 sh evaluate.sh ${CHECKPOINT} mmbench-dev-cn --dynamic
# Test the MMBench-Test-CN
GPUS=8 sh evaluate.sh ${CHECKPOINT} mmbench-test-cn --dynamic
# Test the CCBench-Dev
GPUS=8 sh evaluate.sh ${CHECKPOINT} ccbench-dev --dynamic
```

After the test is completed, a file with a name similar to `results/mmbench_dev_20230712_241224214015.xlsx` will be generated. Please upload these files to the [official server](https://mmbench.opencompass.org.cn/mmbench-submission) to obtain the evaluation scores.

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default                  | Description                                                                                                       |
| ---------------- | ------ | ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`                     | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'mmbench_dev_20230712'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`                  | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`                      | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`                  | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`                  | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
