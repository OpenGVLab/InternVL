# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `LLaVA-Bench`.

For scoring, we use **GPT-4-0613** as the evaluation model.
While the provided code can run the benchmark, we recommend using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for testing this benchmark if you aim to align results with our technical report.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### LLaVA-Bench

Follow the instructions below to prepare the data:

```shell
# Step 1: Download the dataset
cd data/
git clone https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild
cd ../
```

After preparation is complete, the directory structure is:

```shell
data/llava-bench-in-the-wild
├── images
├── answers_gpt4.jsonl
├── bard_0718.jsonl
├── bing_chat_0629.jsonl
├── context.jsonl
├── questions.jsonl
└── README.md
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 1-GPU setup:

```shell
# Step 1: Remove old inference results if exists
rm -rf results/llava_bench_results_review.jsonl

# Step 2: Run the evaluation
torchrun --nproc_per_node=1 eval/llava_bench/evaluate_llava_bench.py --checkpoint ${CHECKPOINT} --dynamic

# Step 3: Scoring the results using gpt-4-0613
export OPENAI_API_KEY="your_openai_api_key"
python -u eval/llava_bench/eval_gpt_review_bench.py \
  --question data/llava-bench-in-the-wild/questions.jsonl \
  --context data/llava-bench-in-the-wild/context.jsonl \
  --rule eval/llava_bench/rule.json \
  --answer-list \
      data/llava-bench-in-the-wild/answers_gpt4.jsonl \
      results/llava_bench_results.jsonl \
  --output \
      results/llava_bench_results_review.jsonl
python -u eval/llava_bench/summarize_gpt_review.py -f results/llava_bench_results_review.jsonl
```

Alternatively, you can run the following simplified command:

```shell
export OPENAI_API_KEY="your_openai_api_key"
GPUS=1 sh evaluate.sh ${CHECKPOINT} llava-bench --dynamic
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default         | Description                                                                                                       |
| ---------------- | ------ | --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`            | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'llava_bench'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`         | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`             | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`         | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`         | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
