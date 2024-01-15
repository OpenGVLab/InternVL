#!/usr/bin/env bash

set -x
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

OUTPUT_DIR=$1
MODEL_NAME=$(basename ${OUTPUT_DIR})

python -m llava.eval.model_vqa \
    --model-path ${OUTPUT_DIR} \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${MODEL_NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${MODEL_NAME}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${MODEL_NAME}.json
