#!/bin/bash

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-256}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

# MODEL="internvl_sft_internvl2_5_8b_dynamic_res_sft_mmmu_o1_241125"
MODEL="internvl_sft_internvl2_pro_dynamic_res_sft_mmmu_o1_241125"

LOG_DIR="logs_sampling/correctness_llm_check"
DATA_DIR="outputs/correctness_mmmu_o1_241125/${MODEL}"
SAVE_DIR="outputs/correctness_llm_check_mmmu_o1_241125/${MODEL}"

model_path="/mnt/petrelfs/share_data/wangweiyun/share_ckpt_hf/Qwen2.5-72B-Instruct"


# set -x

dataset_name="$(basename ${DATA_DIR})"
model_name="$(basename $(dirname ${model_path}))_$(basename ${model_path})"

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
    --job-name "wwy_sampling" \
    -o "${CUR_LOG_DIR}/${dataset_name}.log" \
    -e "${CUR_LOG_DIR}/${dataset_name}.log" \
python -u tools/mmpr_pipeline/internvl_lmdeploy_correctness_postprocess_with_llm.py \
    --judger $model_path \
    --data-dir $DATA_DIR \
    --save-dir $SAVE_DIR \
    --batch-size 8 \
    --tp 8
