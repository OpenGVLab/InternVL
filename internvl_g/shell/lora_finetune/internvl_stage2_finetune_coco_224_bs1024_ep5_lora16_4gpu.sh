set -x

GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-32}


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl_stage2_finetune_coco_364_bs1024_ep5_lora_4gpu'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 32
# batch size per gpu: 32
# gradient accumulation steps: 1
# total batch size: 1024
# epoch: 5
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_stage2_finetune.py \
  --dataset_name 'coco_karpathy_train' \
  --model_name_or_path "./pretrained/InternVL-14B-224px" \
  --output_dir ${OUTPUT_DIR} \
  --overwrite_output_dir True \
  --freeze_model \
  --freeze_vision_model \
  --freeze_qllama \
  --unfreeze_qllama_head \
  --use_backbone_lora 16 \
  --use_qllama_lora 16 \
  --force_image_size 224 \
  --drop_path_rate 0.0 \
  --dataloader_num_workers 2 \
  --pad_to_max_length True \
  --bf16 True \
  --num_train_epochs 5 \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 5 \
  --learning_rate 1e-6 \
  --weight_decay 0.05 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 80 \
  --do_train True \
  --optim adamw_torch \
  --deepspeed "zero_stage3_config.json" \
  --report_to "tensorboard"
