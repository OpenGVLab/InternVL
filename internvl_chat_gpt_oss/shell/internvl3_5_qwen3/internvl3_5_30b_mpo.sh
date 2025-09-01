set -x

export MASTER_PORT=34235
export TF_CPP_MIN_LOG_LEVEL=3
export USE_TCS_LOADER=0
export LAUNCHER=pytorch

# Set the task name
CURRENT_PATH=$(pwd)
PROJECT_NAME=internvl3_5_30b_mpo
TASK_NAME=$(basename "$0")
TASK_NAME="${TASK_NAME%.*}"
echo "TASK_NAME: $TASK_NAME"
echo "PROJECT_NAME: $PROJECT_NAME"

export OUTPUT_DIR=${CURRENT_PATH}/work_dirs/${PROJECT_NAME}/${TASK_NAME}
export TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
export JOBLOG=${OUTPUT_DIR}/training.log

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
BATCH_SIZE=${BATCH_SIZE:-256}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / WORLD_SIZE / NPROC_PER_NODE))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRITON_CACHE_DIR="/dev/shm/triton_wwy/"
export VLLM_CACHE_ROOT="/dev/shm/vllmca_wwy/"

export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

torchrun \
  --node-rank=$RANK \
  --nnodes=$WORLD_SIZE \
  --nproc-per-node=$NPROC_PER_NODE \
  --master-addr=$MASTER_ADDR \
  --master-port=$MASTER_PORT \
internvl/train/internvl_chat_mpo.py \
  --model_name_or_path "OpenGVLab/InternVL3_5-30B-A3B-Instruct" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "${CURRENT_PATH}/shell/data/debug_mpo.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --min_num_frame 8 \
  --max_num_frame 32 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --dataloader_num_workers 16 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 50 \
  --learning_rate 2e-7 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --split_annotations False \
  --do_train True \
  --grad_checkpoint True \
  --gradient_checkpointing True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --use_custom_flash_attn False \
  --report_to "tensorboard" \
  --deepspeed "zero_stage3_config.json" \
  --seed 42 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
