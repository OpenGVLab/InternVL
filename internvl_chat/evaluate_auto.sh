set -x

CHECKPOINT=${1}

ARGS=("$@")

declare -a tasks=( \
    # 'mme' \
    # 'caption-coco' \
    # 'caption-flickr30k' \
    # 'caption-nocaps' \
    # 'vqa-okvqa-val' \
    # 'vqa-textvqa-val' \
    # 'vqa-vizwiz-val' \
    # 'vqa-chartqa-test' \
    # 'vqa-docvqa-val' \
    # 'vqa-ai2d-test' \
    # 'vqa-infovqa-val' \
    # 'vqa-gqa-testdev' \
    # 'scienceqa' \
    'm3cot' \
    # 'pope' \
    # 'amber' \
    # 'tiny_lvlm' \
    # 'mmmu-val' \
    # 'mmvet' \
    # 'mmbench-test-en' \
    # 'mmbench-test-cn' \
    # 'mmvp' \
    # 'ccbench-dev' \
    # 'seed' \
    # 'mathvista-testmini' \
    # 'mmhal' \
    # 'mmhal-4o' \
    # 'objhal' \
)

LOG_DIR="logs_eval"
mkdir -p $LOG_DIR

for ((j=0; j<${#tasks[@]}; j++)); do
    model_path=$CHECKPOINT
    task=${tasks[j]}

    model_name="$(basename $(dirname ${model_path}))_$(basename ${model_path})"
    echo "$(date) ${model_name}_${task}"

    mkdir -p "${LOG_DIR}/${model_name}"

    if [ "${task}" == "vqa-chartqa-test" ]; then
        srun \
            -p VC5 \
            --gres=gpu:8 \
            --ntasks=1 \
            --ntasks-per-node=1 \
            --job-name "wwy_eval" \
            -o "${LOG_DIR}/${model_name}/${task}.log" \
            -e "${LOG_DIR}/${model_name}/${task}.log" \
            --async \
        sh evaluate.sh ${model_path} ${task} --dynamic --max-num 12 "${ARGS[@]:1}"
    elif [ "${task}" == "vqa-infovqa-val" ]; then
        srun \
            -p VC5 \
            --gres=gpu:8 \
            --ntasks=1 \
            --ntasks-per-node=1 \
            --job-name "wwy_eval" \
            -o "${LOG_DIR}/${model_name}/${task}.log" \
            -e "${LOG_DIR}/${model_name}/${task}.log" \
            --async \
        sh evaluate.sh ${model_path} ${task} --dynamic --max-num 24 "${ARGS[@]:1}"
    elif [ "${task}" == "vqa-docvqa-val" ]; then
        srun \
            -p VC5 \
            --gres=gpu:8 \
            --ntasks=1 \
            --ntasks-per-node=1 \
            --job-name "wwy_eval" \
            -o "${LOG_DIR}/${model_name}/${task}.log" \
            -e "${LOG_DIR}/${model_name}/${task}.log" \
            --async \
        sh evaluate.sh ${model_path} ${task} --dynamic --max-num 18 "${ARGS[@]:1}"
    else
        srun \
            -p VC5 \
            --gres=gpu:8 \
            --ntasks=1 \
            --ntasks-per-node=1 \
            --job-name "wwy_eval" \
            -o "${LOG_DIR}/${model_name}/${task}.log" \
            -e "${LOG_DIR}/${model_name}/${task}.log" \
            --async \
        sh evaluate.sh ${model_path} ${task} --dynamic --max-num 6 "${ARGS[@]:1}"
    fi
done
