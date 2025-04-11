#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

PROMPT_VERSION="en_v2"
data_dir="outputs_mpo/correctness_mmpr_v1_2_${PROMPT_VERSION}"
save_dir="outputs_mpo/correctness_mmpr_v1_2_${PROMPT_VERSION}_pairs"

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
    python -u tools/reasoning_data_pipeline/mmpr_data_pipeline_correctness_postprocess.py \
        --data-dir "${data_dir}/${model}/max_tiles_${curr_max_tiles}" \
        --save-dir "${save_dir}/${model}" \
        --answer-fix \
        --force \
        --num-pairs-per-key 15 \
        --max-lines 1200000

done
