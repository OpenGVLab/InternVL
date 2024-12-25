# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MMMU-Pro`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### MMMU-Pro

The evaluation script will automatically download the MMMU-Pro dataset from HuggingFace, and the cached path is `data/MMMU`.

## 🏃 Evaluation Execution

This evaluation script requires `lmdeploy`. If it's not installed, run the following command:

```shell
pip install lmdeploy>=0.5.3 --no-deps
```

To run the evaluation, execute the following command on an 1-GPU setup:

```shell
python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode direct --setting "standard (10 options)" --tp 1
python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode cot --setting "standard (10 options)" --tp 1
python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode direct --setting vision --tp 1
python -u eval/mmmu_pro/evaluate_mmmu_pro.py --model ${CHECKPOINT} --mode cot --setting vision --tp 1
```

Alternatively, you can run the following simplified command:

```shell
GPUS=1 sh evaluate.sh ${CHECKPOINT} mmmu-pro-std10 --tp 1
GPUS=1 sh evaluate.sh ${CHECKPOINT} mmmu-pro-vision --tp 1
```

After the test is complete, run the following command to get the score:

```shell
python eval/mmmu_pro/evaluate.py
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument    | Type  | Default                    | Description                                                                                     |
| ----------- | ----- | -------------------------- | ----------------------------------------------------------------------------------------------- |
| `--model`   | `str` | `'OpenGVLab/InternVL2-8B'` | Specifies the model name to use in the pipeline.                                                |
| `--mode`    | `str` | `'direct'`                 | Defines the operation mode, such as `direct` or `cot`.                                          |
| `--setting` | `str` | `'standard (10 options)'`  | Determines the setting for processing the dataset, such as `standard (10 options)` or `vision`. |
| `--tp`      | `int` | `1`                        | Sets tensor parallelism (TP) for distributing computations across multiple GPUs.                |
