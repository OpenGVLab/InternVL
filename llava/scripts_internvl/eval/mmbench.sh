#!/usr/bin/env bash

set -x
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

OUTPUT_DIR=$1
MODEL_NAME=$(basename ${OUTPUT_DIR})

SPLIT="mmbench_dev_20230712"  # "mmbench_dev_20230712" or "mmbench_test_en_20231003"

python -m llava.eval.model_vqa_mmbench \
    --model-path ${OUTPUT_DIR} \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${MODEL_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${MODEL_NAME}
