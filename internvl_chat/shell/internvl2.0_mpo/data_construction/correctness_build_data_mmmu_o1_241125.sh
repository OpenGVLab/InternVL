#!/bin/bash

data_dir="outputs/correctness_llm_check_mmmu_o1_241125"
save_dir="outputs_pair_data/correctness_mmmu_o1_241125"
model="internvl_sft_internvl2_5_8b_dynamic_res_sft_mmmu_o1_241125"

declare -a max_tiles=( \
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
    python -u tools/mmpr_pipeline/internvl_lmdeploy_correctness_postprocess.py \
        --data-dir "${data_dir}/${model}/max_tiles_${curr_max_tiles}" \
        --save-dir "${save_dir}/${model}" \
        --answer-fix \
        --force \
        --num-pairs-per-key 10 \
        --max-lines 1200000 \
        --use-correctness-cache

done

python -u tools/mmpr_pipeline/internvl_auto_meta.py \
    --data-dir "${save_dir}/${model}" \
