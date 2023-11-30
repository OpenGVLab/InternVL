#!/usr/bin/env bash

set -x
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

OUTPUT_DIR=$1
MODEL_NAME=$(basename ${OUTPUT_DIR})

python -m llava.eval.model_vqa \
    --model-path ${OUTPUT_DIR} \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/${MODEL_NAME}.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/${MODEL_NAME}.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/${MODEL_NAME}.jsonl
