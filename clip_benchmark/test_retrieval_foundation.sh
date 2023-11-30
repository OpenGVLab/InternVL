#!/usr/bin/env bash

set -x

/usr/local/lib/miniconda3/condabin/conda init bash
source activate
conda activate /mnt/afs/user/chenzhe/.conda/envs/husky

cd /mnt/afs/user/chenzhe/workspace/InternVL/benchmark/


export PYTHONPATH=/mnt/afs/user/chenzhe/workspace/petrel-oss-sdk
export PYTHONPATH=$PYTHONPATH:/mnt/afs/user/chenzhe/workspace/InternVL/benchmark
export CUDA_HOME="/usr/local/cuda-11.8/"
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64/:$LD_LIBRARY_PATH"

CUDA_VISIBLE_DEVICES=0 python clip_benchmark/cli.py eval --model_type internvl_qformer_hf_retrieval --language en --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./imagenet/mscoco_captions/ --model internvl_stage2_finetune_coco_364_exp35 --pretrained checkpoint-400 --output result_retrieval_1020.json  &
CUDA_VISIBLE_DEVICES=1 python clip_benchmark/cli.py eval --model_type internvl_qformer_hf_retrieval --language en --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./imagenet/mscoco_captions/ --model internvl_stage2_finetune_coco_364_exp35 --pretrained checkpoint-800 --output result_retrieval_1020.json  &
CUDA_VISIBLE_DEVICES=2 python clip_benchmark/cli.py eval --model_type internvl_qformer_hf_retrieval --language en --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./imagenet/mscoco_captions/ --model internvl_stage2_finetune_coco_364_exp35 --pretrained checkpoint-2400 --output result_retrieval_1020.json  &
CUDA_VISIBLE_DEVICES=3 python clip_benchmark/cli.py eval --model_type internvl_qformer_hf_retrieval --language en --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./imagenet/mscoco_captions/ --model internvl_stage2_finetune_coco_364_exp35 --pretrained checkpoint-2800 --output result_retrieval_1020.json  &
CUDA_VISIBLE_DEVICES=4 python clip_benchmark/cli.py eval --model_type internvl_qformer_hf_retrieval --language en --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./imagenet/mscoco_captions/ --model internvl_stage2_finetune_coco_364_exp35 --pretrained checkpoint-3200 --output result_retrieval_1020.json  &
CUDA_VISIBLE_DEVICES=5 python clip_benchmark/cli.py eval --model_type internvl_qformer_hf_retrieval --language en --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./imagenet/mscoco_captions/ --model internvl_stage2_finetune_coco_364_exp35 --pretrained checkpoint-3600 --output result_retrieval_1020.json  &
CUDA_VISIBLE_DEVICES=6 python clip_benchmark/cli.py eval --model_type internvl_qformer_hf_retrieval --language en --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./imagenet/mscoco_captions/ --model internvl_stage2_finetune_coco_364_exp35 --pretrained checkpoint-4000 --output result_retrieval_1020.json  &
CUDA_VISIBLE_DEVICES=7 python clip_benchmark/cli.py eval --model_type internvl_qformer_hf_retrieval --language en --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./imagenet/mscoco_captions/ --model internvl_stage2_finetune_coco_364_exp35 --pretrained checkpoint-4400 --output result_retrieval_1020.json  &

wait
