#!/bin/bash

data_dir="outputs/correctness_sftv165"
save_dir="outputs_pair_data/correctness_sftv165"
model="internvl_sft_internvl2_8b_dynamic_res_sft_cotv0"

declare -a max_tiles=( \
    "1" \
    "6" \
    # "12" \
    # "18" \
    # "24" \
)

for ((j=0; j<${#max_tiles[@]}; j++)); do
    curr_max_tiles=${max_tiles[j]}
    echo "$(date) ${model} ${curr_max_tiles}"

    srun \
        -p llm_s \
        --gres=gpu:0 \
    python -u tools/mmpr_pipeline/internvl_lmdeploy_correctness_postprocess.py \
        --data-dir "${data_dir}/${model}/max_tiles_${curr_max_tiles}" \
        --extra-data-dir "outputs/correctness_sftv165/Qwen2.5-72B-Instruct/max_tiles_0" \
        --save-dir "${save_dir}/${model}" \
        --answer-fix \
        --force \
        --num-pairs-per-key 5 \
        --max-lines 1200000 \

done

python -u tools/mmpr_pipeline/internvl_auto_meta.py \
    --data-dir "${save_dir}/${model}" \
