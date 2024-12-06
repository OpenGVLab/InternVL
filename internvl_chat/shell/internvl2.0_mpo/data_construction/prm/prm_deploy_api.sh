PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

CUR_LOG_DIR="logs_sampling/api"

mkdir -p ${CUR_LOG_DIR}

model="work_dirs/internvl_sft/internvl2_5_8b_dynamic_res_sft_mmmu_o1_241125"
model_name="$(basename $(dirname ${model}))_$(basename ${model})"

set -x

srun -p ${PARTITION} \
    --gres=gpu:${GPUS_PER_NODE} \
    --nodes=${NODES} \
    --ntasks=$((GPUS / GPUS_PER_TASK)) \
    --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
    --job-name "wwy_api" \
    -o "${CUR_LOG_DIR}/${model_name}_api.log" \
    -e "${CUR_LOG_DIR}/${model_name}_api.log" \
lmdeploy serve api_server \
    ${model} \
    --cache-max-entry-count 0.1 \
    --session-len 16384 \
    --tp ${GPUS_PER_TASK} \
    --enable-prefix-caching \
    --server-port 23333
