# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MathVista`.

For scoring, we use **GPT-4-0613** as the evaluation model.
While the provided code can run the benchmark, we recommend using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for testing this benchmark if you aim to align results with our technical report.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### MathVista

The evaluation script will automatically download the MathVista dataset from HuggingFace, and the cached path is `data/MathVista`.

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
export OPENAI_API_KEY="your_openai_api_key"
# Test the testmini set
torchrun --nproc_per_node=8 eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --dynamic --datasets MathVista_testmini
# Test the test set
torchrun --nproc_per_node=8 eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --dynamic --datasets MathVista_test
```

Alternatively, you can run the following simplified command:

```shell
export OPENAI_API_KEY="your_openai_api_key"
# Test the testmini set
GPUS=8 sh evaluate.sh ${CHECKPOINT} mathvista-testmini --dynamic
# Test the test set
GPUS=8 sh evaluate.sh ${CHECKPOINT} mathvista-test --dynamic
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default                | Description                                                                                                       |
| ---------------- | ------ | ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`                   | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'MathVista_testmini'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`                | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`                    | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`                | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`                | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
