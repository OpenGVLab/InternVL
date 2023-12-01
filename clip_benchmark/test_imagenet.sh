set -x

alias s1a='srun -p INTERN4 -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto'

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification"  --dataset "imagenet1k" --dataset_root ./imagenet/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" \
    --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "it" \
    --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "jp" \
    --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "ar" \
    --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "imagenetv2" --dataset_root ./imagenet/imagenetv2/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "imagenet_sketch" --dataset_root ./imagenet/sketch/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "imagenet-a" --dataset_root ./imagenet/imagenet-a/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "imagenet-r" --dataset_root ./imagenet/imagenet-r/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" \
    --task "zeroshot_classification" --dataset "objectnet" --dataset_root ./imagenet/objectnet-1.0/ \
    --model ${MODEL} --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json
