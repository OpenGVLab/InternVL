#!/bin/bash

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-256}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

LOG_DIR="logs_sampling/correctness_mmmu_o1_241125"


# model_path="work_dirs/internvl_sft/internvl2_5_8b_dynamic_res_sft_mmmu_o1_241125"
model_path="work_dirs/internvl_sft/internvl2_pro_dynamic_res_sft_mmmu_o1_241125"


declare -a datasets=( \
    'outputs/correctness_prompt_mmmu_o1_241125/mmmu_similar_images_qa_cot_training_21092.jsonl,6' \
    'outputs/correctness_prompt_mmmu_o1_241125/test_dev_set_cot_training_241125_4535.jsonl,6' \
    'outputs/correctness_prompt_mmmu_o1_241125/val_new_question_cot_training_241125_2253.jsonl,6' \
)

# set -x

for ((i=0; i<${#datasets[@]}; i++)); do

    dataset="$(echo ${datasets[i]} | awk -F',' '{print $1}')"
    max_num="$(echo ${datasets[i]} | awk -F',' '{print $2}')"
    dataset_name="$(basename ${dataset})"
    model_name="$(basename $(dirname ${model_path}))_$(basename ${model_path})"

    echo "$(date) ${dataset} ${max_num}"

    CUR_LOG_DIR="${LOG_DIR}/${model_name}"
    mkdir -p "$CUR_LOG_DIR"

    wc -l ${dataset}

    srun -p ${PARTITION} \
        --gres=gpu:${GPUS_PER_NODE} \
        --nodes=${NODES} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        --quotatype=${QUOTA_TYPE} \
        --job-name "wwy_sampling" \
        -o "${CUR_LOG_DIR}/${dataset_name}_max_tiles_${max_num}.log" \
        -e "${CUR_LOG_DIR}/${dataset_name}_max_tiles_${max_num}.log" \
    python -u tools/mmpr_pipeline/internvl_lmdeploy_correctness.py \
        --checkpoint $model_path \
        --prompt-path $dataset \
        --out-dir "outputs/correctness_mmmu_o1_241125" \
        --batch-size 8 \
        --num-workers 8 \
        --num-return-sequences 32 \
        --top-k 50 \
        --temperature 1.0 \
        --dynamic \
        --max-num ${max_num} \
        --sample-max-num 30000 \
        --tp 8

done
