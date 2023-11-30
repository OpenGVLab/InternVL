set -x

PARTITION=${PARTITION:-'INTERN4'}
alias s1a="srun -p ${PARTITION} -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_retrieval" \
    --dataset "multilingual_mscoco_captions" --dataset_root ./data/mscoco_captions --model internvl_g_retrieval_hf \
    --pretrained ./pretrained/internvl_14b_224px --output result_g.json --language=en

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_retrieval" \
    --dataset "multilingual_mscoco_captions" --dataset_root ./data/mscoco_captions --model internvl_g_retrieval_hf \
    --pretrained ./pretrained/internvl_14b_224px --output result_g.json --language=es

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
    --dataset "multilingual_mscoco_captions" --dataset_root ./data/mscoco_captions --model internvl_g_retrieval_hf \
    --pretrained ./pretrained/internvl_14b_224px --output result_g.json --language=fr

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
    --dataset "multilingual_mscoco_captions" --dataset_root ./data/mscoco_captions --model internvl_g_retrieval_hf \
    --pretrained ./pretrained/internvl_14b_224px --output result_g.json --language=zh

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
    --dataset "multilingual_mscoco_captions" --dataset_root ./data/mscoco_captions --model internvl_g_retrieval_hf \
    --pretrained ./pretrained/internvl_14b_224px --output result_g.json --language=it

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
    --dataset "multilingual_mscoco_captions" --dataset_root ./data/mscoco_captions --model internvl_g_retrieval_hf \
    --pretrained ./pretrained/internvl_14b_224px --output result_g.json --language=ko

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
    --dataset "multilingual_mscoco_captions" --dataset_root ./data/mscoco_captions --model internvl_g_retrieval_hf \
    --pretrained ./pretrained/internvl_14b_224px --output result_g.json --language=ru

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
    --dataset "multilingual_mscoco_captions" --dataset_root ./data/mscoco_captions --model internvl_g_retrieval_hf \
    --pretrained ./pretrained/internvl_14b_224px --output result_g.json --language=jp
