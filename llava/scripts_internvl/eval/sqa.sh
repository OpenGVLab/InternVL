#!/usr/bin/env bash

set -x
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

OUTPUT_DIR=$1
MODEL_NAME=$(basename ${OUTPUT_DIR})

python -m llava.eval.model_vqa_science \
    --model-path ${OUTPUT_DIR} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${MODEL_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${MODEL_NAME}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${MODEL_NAME}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${MODEL_NAME}_result.json
