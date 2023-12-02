set -x

alias s1a='srun -p INTERN4 -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto'

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "birdsnap" --dataset_root ./data/birdsnap/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "cifar10" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "cifar100" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "food101" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "sun397" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "cars" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "fgvc_aircraft" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "dtd" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "pets" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "caltech101" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "mnist" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "stl10" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "eurosat" --dataset_root ./data/EuroSAT/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "gtsrb" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "country211" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "pcam" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "renderedsst2" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "fer2013" --dataset_root ./data/fer2013 --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "voc2007" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "vtab/flowers" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json

s1a python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_classification" \
    --dataset "vtab/resisc45" --dataset_root ./data/ --model internvl_c_classification \
    --pretrained ./pretrained/internvl_c_13b_224px.pth --output result.json
