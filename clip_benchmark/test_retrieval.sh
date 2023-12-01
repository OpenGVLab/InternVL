
set -x

MODEL=$1
PRETRAINED=$2
MODEL_TYPE=${MODEL_TYPE:-'internvl_clip_retrieval'}
#MODEL_TYPE=${MODEL_TYPE:-'internvl_qformer_hf_retrieval'}
#MODEL_TYPE=${MODEL_TYPE:-'internvl_clip_hf_retrieval'}

alias s1a='srun -p INTERN4 -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto'
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_retrieval" --dataset "flickr30k" --dataset_root ./data/flickr30k --model ${MODEL} --pretrained ${PRETRAINED} --output result_retrieval_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "en" --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./data/mscoco_captions --model ${MODEL} --pretrained ${PRETRAINED} --output result_retrieval_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "cn" --task "zeroshot_retrieval" --dataset "flickr30k" --dataset_root ./data/flickr30k --model ${MODEL} --pretrained ${PRETRAINED} --output result_retrieval_${MODEL}_${PRETRAINED}.json
s1a --async python3 clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} --language "cn" --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./data/mscoco_captions --model ${MODEL} --pretrained ${PRETRAINED} --output result_retrieval_${MODEL}_${PRETRAINED}.json
