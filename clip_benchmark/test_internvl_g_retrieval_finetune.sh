set -x

PARTITION=${PARTITION:-'INTERN4'}
alias s1a="srun -p ${PARTITION} -N 1 --gres=gpu:1 --cpus-per-task 10 --quotatype=auto"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-400 \
     --output result_g.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-500 \
     --output result_g.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-600 \
     --output result_g.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-700 \
     --output result_g.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-800 \
     --output result_g.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-900 \
     --output result_g.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-1000 \
     --output result_g.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-1100 \
     --output result_g.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-1200 \
     --output result_g.json

s1a --async python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/checkpoint-1300 \
     --output result_g.json
