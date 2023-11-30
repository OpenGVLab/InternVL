
set -x

MODEL=$1
PRETRAINED=$2
MODEL_TYPE=${MODEL_TYPE:-'internvl_clip_classification'}

alias s1a='srun -p VC2 -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto'
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json --save_clf "vit_6b_head.pth"
