
set -x

MODEL=$1
PRETRAINED=$2
MODEL_TYPE=${MODEL_TYPE:-'internvl_clip_classification'}

export http_proxy=http://chenzhe1:Abc88784989@10.1.8.50:33128/
export https_proxy=http://chenzhe1:Abc88784989@10.1.8.50:33128/
export HTTP_PROXY=http://chenzhe1:Abc88784989@10.1.8.50:33128/
export HTTPS_PROXY=http://chenzhe1:Abc88784989@10.1.8.50:33128/

alias s1a='srun -p INTERN2 -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto'
s1a python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "birdsnap" --dataset_root ./imagenet/birdsnap/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "cifar10" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "cifar100" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "food101" --dataset_root ./imagenet/food101 --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "sun397" --dataset_root ./imagenet/sun397 --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "cars" --dataset_root ./imagenet/cars --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "fgvc_aircraft" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "dtd" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "pets" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "caltech101" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "mnist" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "stl10" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "eurosat" --dataset_root ./imagenet/EuroSAT/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "gtsrb" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "country211" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "pcam" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "renderedsst2" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "fer2013" --dataset_root ./imagenet/fer2013 --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "voc2007" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "vtab/flowers" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
#s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_classification" --dataset "vtab/resisc45" --dataset_root ./imagenet/ --model ${MODEL} --pretrained ${PRETRAINED} --output result_vtab_${MODEL}_${PRETRAINED}.json
