#!/bin/bash

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-256}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

PROMPT_VERSION="en_v2"
CHECKPOINT="OpenGVLab/InternVL3-8B"

LOG_DIR="logs_sampling/correctness_mmpr_v1_2_${PROMPT_VERSION}"
OUTPUT_DIR="outputs_mpo/correctness_mmpr_v1_2_${PROMPT_VERSION}"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

if [[ "$CHECKPOINT" =~ 38B ]]; then
    export TP="2"
    export AUTO_SPLIT="1"
elif [[ "$CHECKPOINT" =~ 78B ]]; then
    export TP="4"
    export AUTO_SPLIT="1"
else
    export AUTO_SPLIT="0"
fi

echo "${CHECKPOINT}, ${AUTO_SPLIT}"

# NOTE: you can download MMPR-v1.2-prompts from: https://huggingface.co/datasets/OpenGVLab/MMPR-v1.2-prompts

declare -a datasets=( \
    'MMPR-v1.2-prompts/correctness_prompts/CLEVR_math_en_20240402_extracted.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/geos_en_20240402_extracted.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/geometry3k_en_20240402_extracted.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/scienceqa_multi_choice_en_20240402_extracted.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/m3cot_train_extracted.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/ai2d_train_12k_en_20240410_extracted.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/SROIE_information_extraction_multi_turn_20240620_extracted.jsonl,12' \
    'MMPR-v1.2-prompts/correctness_prompts/chartqa_trainval_30k_w_csv_en_20240402_extracted.jsonl,12' \
    'MMPR-v1.2-prompts/correctness_prompts/docvqa_train_56k_en_20240402_extracted.jsonl,18' \
    'MMPR-v1.2-prompts/correctness_prompts/infographics_20240403_qa_20240407_v2_extracted.jsonl,24' \
    'MMPR-v1.2-prompts/correctness_prompts/mapqa_suv_en_20240402_extracted.jsonl,12' \
    'MMPR-v1.2-prompts/correctness_prompts/figureqa_en_20240402_extracted.jsonl,12' \
    'MMPR-v1.2-prompts/correctness_prompts/dvqa_en_20240402_extracted_int_only.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/iconqa_train_extracted.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/geometry3k_en_20240402_extracted_open_ended_only.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/geoqa+_en_20240402_extracted_open_ended_only.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/geos_en_20240402_extracted_open_ended_only.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/unigeo_calc_en_20240402_extracted_open_ended_only.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/geomverse_extracted.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/geo170k_extracted_full.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/geoqa+_extracted_en_version.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/koniq10k_en_20240403.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/study_com_en_20240619.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_function_abs.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_function_cos.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_function_log.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_function_poly.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_function_sin.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_function_tan.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/super_clevr_en_20240402_int.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/super_clevr_en_20240402_yorn.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/vqav2_en_20240402_int.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/cocorem_exist_yorn_en_20241016.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_geo_depth0_text_dominant_vision_dominant_en.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_geo_depth1_text_dominant_vision_dominant_en.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_geo_depth2_text_dominant_vision_dominant_en.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/mavis_geo_depth3_text_dominant_vision_dominant_en.jsonl,6' \
    'MMPR-v1.2-prompts/correctness_prompts/nlvr2_en_20240910.jsonl,1' \
    'MMPR-v1.2-prompts/correctness_prompts/MathV360K_prompts.jsonl,6' \
)

# set -x

for ((i=0; i<${#datasets[@]}; i++)); do

    dataset="$(echo ${datasets[i]} | awk -F',' '{print $1}')"
    max_num="$(echo ${datasets[i]} | awk -F',' '{print $2}')"
    dataset_name="$(basename ${dataset})"
    model_name="$(basename $(dirname ${CHECKPOINT}))_$(basename ${CHECKPOINT})"

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
    python -u tools/reasoning_data_pipeline/mmpr_data_pipeline_correctness.py \
        --checkpoint $CHECKPOINT \
        --prompt-path $dataset \
        --out-dir $OUTPUT_DIR \
        --num-workers 8 \
        --batch-size 8 \
        --vit-batch-size 8 \
        --dynamic \
        --max-num ${max_num} \
        --session-len 16384 \
        --top-k 50 \
        --temperature 0.7 \
        --max-new-tokens 4096 \
        --num-return-sequences 32 \
        --sample-start-idx 78 \
        --sample-max-num 7000 \
        --prompt-version ${PROMPT_VERSION}

done
