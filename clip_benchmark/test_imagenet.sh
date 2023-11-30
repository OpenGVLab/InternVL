
set -x

MODEL=$1
PRETRAINED=$2
MODEL_TYPE=${MODEL_TYPE:-'internvl_clip_classification'}

alias s1a='srun -p INTERN3 -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto'
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "cn" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "it" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "jp" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "ar" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "imagenetv2" --dataset_root ./imagenet/imagenetv2/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "imagenet_sketch" --dataset_root ./imagenet/sketch/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "imagenet-a" --dataset_root ./imagenet/imagenet-a/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "imagenet-r" --dataset_root ./imagenet/imagenet-r/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "objectnet" --dataset_root ./imagenet/objectnet-1.0/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_imagenet_${MODEL}_${PRETRAINED}.json
