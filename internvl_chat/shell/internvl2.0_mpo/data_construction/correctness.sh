#!/bin/bash

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-256}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

LOG_DIR="logs_sampling/correctness"

model_path="/mnt/petrelfs/wangweiyun/workspace_wwy/InternVL/internvl_chat/work_dirs/internvl_sft/internvl2_5_8b_dynamic_res_sft_cotv4"

declare -a datasets=( \
    'outputs/correctness_prompt_mmpr/CLEVR_math_en_20240402_extracted.jsonl,6' \
    'outputs/correctness_prompt_mmpr/geos_en_20240402_extracted.jsonl,6' \
    'outputs/correctness_prompt_mmpr/geometry3k_en_20240402_extracted.jsonl,6' \
    'outputs/correctness_prompt_mmpr/scienceqa_multi_choice_en_20240402_extracted.jsonl,6' \
    'outputs/correctness_prompt_mmpr/m3cot_train_extracted.jsonl,6' \
    'outputs/correctness_prompt_mmpr/ai2d_train_12k_en_20240410_extracted.jsonl,6' \
    #
    'outputs/correctness_prompt_mmpr/SROIE_information_extraction_multi_turn_20240620_extracted.jsonl,12' \
    'outputs/correctness_prompt_mmpr/chartqa_trainval_30k_w_csv_en_20240402_extracted.jsonl,12' \
    'outputs/correctness_prompt_mmpr/docvqa_train_56k_en_20240402_extracted.jsonl,18' \
    'outputs/correctness_prompt_mmpr/infographics_20240403_qa_20240407_v2_extracted.jsonl,24' \
    #
    'outputs/correctness_prompt_mmpr/mapqa_suv_en_20240402_extracted.jsonl,12' \
    'outputs/correctness_prompt_mmpr/figureqa_en_20240402_extracted.jsonl,12' \
    #
    'outputs/correctness_prompt_mmpr/dvqa_en_20240402_extracted_int_only.jsonl,6' \
    'outputs/correctness_prompt_mmpr/iconqa_train_extracted.jsonl,6' \
    #
    'outputs/correctness_prompt_mmpr/geometry3k_en_20240402_extracted_open_ended_only.jsonl,6' \
    'outputs/correctness_prompt_mmpr/geoqa+_en_20240402_extracted_open_ended_only.jsonl,6' \
    'outputs/correctness_prompt_mmpr/geos_en_20240402_extracted_open_ended_only.jsonl,6' \
    'outputs/correctness_prompt_mmpr/unigeo_calc_en_20240402_extracted_open_ended_only.jsonl,6' \
    'outputs/correctness_prompt_mmpr/geomverse_extracted.jsonl,6' \
    'outputs/correctness_prompt_mmpr/geo170k_extracted_full.jsonl'
    'outputs/correctness_prompt_mmpr/geoqa+_extracted_en_version.jsonl,6' \
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
        --job-name "wwy_sampling" \
        -o "${CUR_LOG_DIR}/${dataset_name}_max_tiles_${max_num}.log" \
        -e "${CUR_LOG_DIR}/${dataset_name}_max_tiles_${max_num}.log" \
    python -u tools/mmpr_pipeline/internvl_lmdeploy_correctness.py \
        --checkpoint $model_path \
        --prompt-path $dataset \
        --out-dir "outputs/correctness_mmpr" \
        --batch-size 8 \
        --num-workers 8 \
        --num-return-sequences 32 \
        --top-k 50 \
        --temperature 1.0 \
        --dynamic \
        --max-num ${max_num} \
        --sample-max-num 30000 \
        --tp 8

done
