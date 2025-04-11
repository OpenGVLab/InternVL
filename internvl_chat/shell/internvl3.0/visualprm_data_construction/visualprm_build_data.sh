#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

PROMPT_VERSION="en_v2"
data_dir="outputs_prm/visualprm_v1_1_${PROMPT_VERSION}_raw"
save_dir="outputs_prm/visualprm_v1_1_${PROMPT_VERSION}_conv"

model="OpenGVLab_InternVL3-8B"

declare -a max_tiles=( \
    "1" \
    "6" \
    "12" \
    "18" \
    "24" \
)

for ((j=0; j<${#max_tiles[@]}; j++)); do
    curr_max_tiles=${max_tiles[j]}
    echo "$(date) ${model} ${curr_max_tiles}"

    srun \
        -p Intern5 \
        --gres=gpu:0 \
    python -u tools/reasoning_data_pipeline/visualprm_data_pipeline_postprocess.py \
        --data-dir "${data_dir}/${model}/max_tiles_${curr_max_tiles}" \
        --save-dir "${save_dir}/${model}" \
        --mc-threshold 0.0

done
