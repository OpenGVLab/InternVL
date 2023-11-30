set -x

PARTITION=${PARTITION:-'INTERN4'}
alias s1a="srun -p ${PARTITION} -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_c_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10 \
     --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10 \
     --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_c_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickr_364_bs1024_ep10 \
     --output result.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickr_364_bs1024_ep10 \
     --output result.json
