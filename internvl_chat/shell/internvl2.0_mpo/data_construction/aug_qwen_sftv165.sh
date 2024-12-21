#!/bin/bash

PARTITION=${PARTITION:-"VC5"}
GPUS=${GPUS:-256}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"spot"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

LOG_DIR="logs_sampling/aug_qwen_sftv165"
model_path="ckpt/Qwen2.5-72B-Instruct"

declare -a datasets=( \
    # '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/CaraJ/MAVIS-Function/mavis_function_abs.jsonl' \
    # '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/CaraJ/MAVIS-Function/mavis_function_cos.jsonl' \
    # '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/CaraJ/MAVIS-Function/mavis_function_log.jsonl' \
    # '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/CaraJ/MAVIS-Function/mavis_function_poly.jsonl' \
    '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/CaraJ/MAVIS-Function/mavis_function_sin.jsonl' \
    '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/CaraJ/MAVIS-Function/mavis_function_tan.jsonl' \
)

for ((i=0; i<${#datasets[@]}; i++)); do

    dataset=${datasets[i]}
    dataset_name="$(basename ${dataset})"
    model_name="$(basename ${model_path})"

    echo "$(date) ${dataset}"

    CUR_LOG_DIR="${LOG_DIR}/${model_name}"
    mkdir -p "$CUR_LOG_DIR"

    srun -p ${PARTITION} \
        --gres=gpu:${GPUS_PER_NODE} \
        --nodes=${NODES} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        --quotatype=${QUOTA_TYPE} \
        -o "${CUR_LOG_DIR}/${dataset_name}.log" \
        -e "${CUR_LOG_DIR}/${dataset_name}.log" \
    python -u tools/mmpr_pipeline/internvl_lmdeploy_rationale_aug.py \
        --checkpoint $model_path \
        --data-path $dataset \
        --save-dir "outputs/correctness_sftv165/${model_name}/max_tiles_0" \
        --bsz 8 \
        --max-lines 50000

        sleep 1

done
