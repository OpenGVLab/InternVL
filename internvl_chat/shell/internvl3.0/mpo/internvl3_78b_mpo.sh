set -x

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-512}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRITON_CACHE_DIR="/tmp/triton_wwy/"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/internvl_chat_v3_mpo/Internvl3-78B'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# NOTE: you can download MMPR-v1.2 from: https://huggingface.co/datasets/OpenGVLab/MMPR-v1.2
# NOTE: In our experiment, the checkpoint saved at step 400 yields the best performance.

srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python -u internvl/train/internvl_chat_mpo.py \
  --model_name_or_path "OpenGVLab/InternVL3-78B-Instruct" \
  --conv_style "internvl2_5" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "MMPR-v1.2/meta.json" \
  --overwrite_output_dir False \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 100 \
  --learning_rate 2e-7 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config_100b_1e7_offload.json" \
  --report_to "tensorboard" \
  --loss_type sigmoid,bco_pair \
  --sigmoid_loss_weight 0.8 \
  --bco_pair_loss_weight 0.2 \
  --rpo_alpha 1 \
  --use_liger True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
