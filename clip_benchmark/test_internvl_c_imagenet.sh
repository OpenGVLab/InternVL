set -x

PARTITION=${PARTITION:-'INTERN4'}
alias s1a="srun -p ${PARTITION} -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./data/imagenet-1k/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" \
    --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./data/imagenet-1k/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "it" \
    --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./data/imagenet-1k/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "jp" \
    --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./data/imagenet-1k/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "ar" \
    --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./data/imagenet-1k/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "imagenetv2" --dataset_root ./data/imagenetv2/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "imagenet_sketch" --dataset_root ./data/imagenet-sketch/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "imagenet-a" --dataset_root ./data/imagenet-a/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "imagenet-r" --dataset_root ./data/imagenet-r/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "objectnet" --dataset_root ./data/objectnet-1.0/ \
    --model internvl_c_classification --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json
