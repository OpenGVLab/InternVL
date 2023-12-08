#!/usr/bin/env bash

set -x

/usr/local/lib/miniconda3/condabin/conda init bash
source activate
conda activate /mnt/afs/user/chenzhe/.conda/envs/husky

cd /mnt/afs/user/chenzhe/workspace/InternVL/Husky2/eval/mme/

which java

export PYTHONPATH=/mnt/afs/user/chenzhe/workspace/petrel-oss-sdk
export PYTHONPATH=$PYTHONPATH:/mnt/afs/user/chenzhe/workspace/InternVL/Husky2
export CUDA_HOME="/usr/local/cuda-11.8/"
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64/:$LD_LIBRARY_PATH"

CHECKPOINT=${1}
DIRNAME=`basename ${CHECKPOINT}`
python eval.py --template "vicuna_v1.1" --model_path ${CHECKPOINT}
python calculation.py --results_dir ${DIRNAME}
