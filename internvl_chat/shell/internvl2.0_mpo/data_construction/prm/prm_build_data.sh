#!/bin/bash

data_dir="outputs/prm_mmpr"
save_dir="outputs_pair_data/prm_mmpr"
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
        -p INTERN2 \
        --gres=gpu:0 \
    python -u tools/internvlo1_pipeline/internvl_o1_prm_postprocess.py \
        --overwrite \
        --data-dir "${data_dir}/${model}/max_tiles_${curr_max_tiles}" \
        --save-dir "${save_dir}/${model}" \
        --num-pairs-per-tree 32

done

python -u tools/mmpr_pipeline/internvl_auto_meta.py \
    --data-dir "${save_dir}/${model}" \
    --force\
    --suffix "_process" \
