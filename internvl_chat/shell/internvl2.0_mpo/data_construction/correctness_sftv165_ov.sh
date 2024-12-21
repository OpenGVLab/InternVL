#!/bin/bash

PARTITION=${PARTITION:-"VC5"}
GPUS=${GPUS:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"spot"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

LOG_DIR="logs_sampling/correctness_sftv165"


model_path="/mnt/petrelfs/wangweiyun/workspace_wwy/InternVL-RL-DPO/internvl_chat_dev/work_dirs/internvl_sft/internvl2_8b_dynamic_res_sft_cotv0"


declare -a datasets=( \
    'outputs/correctness_prompt_sftv165/nlvr2_en_20240910.jsonl,1' \
    # 'outputs/correctness_prompt_sftv165/study_com_en_20240619.jsonl,1' \
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
        -o "${CUR_LOG_DIR}/${dataset_name}_max_tiles_${max_num}_ov.log" \
        -e "${CUR_LOG_DIR}/${dataset_name}_max_tiles_${max_num}_ov.log" \
    python -u tools/mmpr_pipeline/internvl_lmdeploy_correctness.py \
        --checkpoint $model_path \
        --prompt-path $dataset \
        --out-dir "outputs/correctness_sftv165" \
        --batch-size 4 \
        --vit-batch-size 1 \
        --num-workers 8 \
        --num-return-sequences 32 \
        --top-k 50 \
        --temperature 1.0 \
        --max-new-tokens 2048 \
        --multi-image \
        --max-num ${max_num} \
        --sample-max-num 30000 \
        --tp 8

done
