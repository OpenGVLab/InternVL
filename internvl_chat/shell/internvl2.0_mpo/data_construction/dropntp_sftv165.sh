#!/bin/bash

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-256}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"spot"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

START_RATIO=${START_RATIO:-"0.50"}
LOG_DIR="logs_sampling/dropntp_sftv165"
OUTPUT_DIR="outputs_pair_data/dropntp_sftv165"

model_path="/mnt/petrelfs/share_data/wangweiyun/share_internvl/InternVL2_5-8B"

declare -a datasets=( \
    # 'outputs/dropntp_sftv165/artwork_part1_en_20240628.jsonl,6' \
    # 'outputs/dropntp_sftv165/sharegpt4o_longcap_en_20240819.jsonl,6' \
    # 'outputs/dropntp_sftv165/sharegpt4o_longcap_zh_20240819.jsonl,6' \
    # 'outputs/dropntp_sftv165/k12_merge_ab_zh_20240812.jsonl,6' \
    # 'outputs/dropntp_sftv165/img_diff_object_replacement_en_20240902.jsonl,6' \
    # 'outputs/dropntp_sftv165/img_diff_object_removal_en_20240902.jsonl,6' \
    # 'outputs/dropntp_sftv165/birds_to_words_en_20240910.jsonl,6' \
    # 'outputs/dropntp_sftv165/private_schedual_extract_zh_20241102.jsonl,6' \
    'outputs/dropntp_sftv165/inat_train2018_merge_en_20240811.jsonl,6' \
    'outputs/dropntp_sftv165/inat_train2018_merge_gpt4o_en_20240819.jsonl,6' \
    'outputs/dropntp_sftv165/spot_the_diff_en_20240910.jsonl,6' \
)

set -x

max_num=6
model_name="$(basename $(dirname ${model_path}))_$(basename ${model_path})"

for ((i=0; i<${#datasets[@]}; i++)); do

    dataset="$(echo ${datasets[i]} | awk -F',' '{print $1}')"
    max_num="$(echo ${datasets[i]} | awk -F',' '{print $2}')"
    dataset_name="$(basename ${dataset})"
    model_name="$(basename $(dirname ${model_path}))_$(basename ${model_path})"

    echo "$(date) ${dataset} ${max_num} ${model_name}"

    CUR_LOG_DIR="${LOG_DIR}/${model_name}_wo_image"
    mkdir -p "$CUR_LOG_DIR"

    srun -p ${PARTITION} \
        --gres=gpu:${GPUS_PER_NODE} \
        --nodes=${NODES} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        --quotatype=${QUOTA_TYPE} \
        -e "${CUR_LOG_DIR}/${dataset_name}_max_tiles_${max_num}_${START_RATIO}.log" \
        -o "${CUR_LOG_DIR}/${dataset_name}_max_tiles_${max_num}_${START_RATIO}.log" \
    python -u tools/mmpr_pipeline/internvl_lmdeploy_dropout_ntp.py \
        --checkpoint $model_path \
        --prompt-path $dataset \
        --out-dir "${OUTPUT_DIR}/max_tiles_${max_num}_wo_image_${START_RATIO}/${model_name}/raw" \
        --batch-size 1 \
        --num-workers 8 \
        --num-return-sequences 1 \
        --top-k 50 \
        --temperature 1.0 \
        --dynamic \
        --max-num ${max_num} \
        --sample-max-num 500000 \
        --tp 8 \
        --start-ratio ${START_RATIO}

done

python -u tools/mmpr_pipeline/internvl_auto_meta.py \
    --data-dir "${OUTPUT_DIR}/max_tiles_${max_num}_wo_image_${START_RATIO}/${model_name}" \
    --suffix "_sr${START_RATIO}_wo_image" \
