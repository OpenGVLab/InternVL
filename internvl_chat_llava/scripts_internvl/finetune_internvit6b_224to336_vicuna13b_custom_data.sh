#!/usr/bin/env bash

set -x

PARTITION=INTERN2
GPUS=${GPUS:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE="reserved"
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
OUTPUT_DIR='work_dirs/finetune_internvit6b_224to336_vicuna13b_custom_data'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

echo "Begin running..."
# number of gpus: 128
# batch size per gpu: 2
# gradient accumulation steps: 2
# total batch size: 512
# epoch: 1
srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u llava/train/train_mem_custom.py \
    --deepspeed ./scripts/zero1.json \
    --model_name_or_path ./pretrained/vicuna-13b-v1.5 \
    --version v1 \
    --meta_path ./scripts_internvl/meta/custom_data.json \
    --vision_tower ./pretrained/InternViT-6B-224px \
    --pretrain_mm_mlp_adapter ./work_dirs/pretrain_internvit6b_224to336_vicuna13b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -4 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to "tensorboard" \
    | tee ${OUTPUT_DIR}/train.log
