set -x

export VIT_LAYER_DECAY_RATE=0.9
export QLLAMA_LAYER_DECAY_RATE=0.9

PARTITION=${PARTITION:-"VC2"}
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}


export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# number of gpus: 32
# batch size per gpu: 32
# gradient accumulation steps: 1
# total batch size: 1024
# epoch: 10
srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u internvl/train/internvl_stage2_finetune.py \
  --dataset_name 'flickr30k_cn_train' \
  --model_name_or_path "./pretrained/InternVL-14B-224px" \
  --output_dir "./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10" \
  --overwrite_output_dir True \
  --force_image_size 364 \
  --drop_path_rate 0.3 \
  --use_custom_trainer \
  --dataloader_num_workers 2 \
  --pad_to_max_length True \
  --bf16 True \
  --num_train_epochs 10 \
  --per_device_train_batch_size 32 \
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
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard"
