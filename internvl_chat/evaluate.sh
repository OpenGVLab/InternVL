set -x

CHECKPOINT=${1}
DATASET=${2}
TEMPLATE=${3}
CHECKPOINT="$(pwd)/${CHECKPOINT}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "CHECKPOINT: ${CHECKPOINT}"

if  [ ${DATASET} == "mme" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  python eval.py --template ${TEMPLATE} --model-path ${CHECKPOINT}
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "caption" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE}
fi

if  [ ${DATASET} == "caption-coco" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets coco
fi

if  [ ${DATASET} == "caption-flickr30k" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets flickr30k
fi

if  [ ${DATASET} == "caption-nocaps" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets nocaps
fi

if [ ${DATASET} == "vqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE}
fi

if [ ${DATASET} == "vqa-okvqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets okvqa_val
fi

if [ ${DATASET} == "vqa-textvqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets textvqa_val_ocr
fi

if [ ${DATASET} == "vqa-vizwiz" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets vizwiz_val
fi

if [ ${DATASET} == "vqa-vizwiz-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets vizwiz_test
fi

if [ ${DATASET} == "vqa-vqav2-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets vqav2_testdev
fi

if [ ${DATASET} == "vqa-ai2d" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets ai2diagram_test
fi

if [ ${DATASET} == "vqa-vqav2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets vqav2_val
fi

if [ ${DATASET} == "vqa-gqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE} --datasets gqa_testdev_llava
fi

if [ ${DATASET} == "refcoco" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE}
fi

if [ ${DATASET} == "llava-bench" ]; then
    python eval/llava_bench/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE}
fi

if [ ${DATASET} == "pope" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope --template ${TEMPLATE}
fi

if [ ${DATASET} == "tiny_lvlm" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/tiny_lvlm/evaluate_lvlm.py --checkpoint ${CHECKPOINT} --datasets updated_datasets --template ${TEMPLATE}
fi

if [ ${DATASET} == "mmvet" ]; then
    python eval/mmvet/evaluate_mmvet.py --checkpoint ${CHECKPOINT} --datasets mmvet --template ${TEMPLATE}
fi

if [ ${DATASET} == "mmbench" ]; then
    cd eval/mmbench/
    python eval.py --checkpoint ${CHECKPOINT} --template ${TEMPLATE}
fi

if [ ${DATASET} == "cmmmu" ]; then
  CUDA_VISIBLE_DEVICES=0 python eval/cmmmu/evaluate_cmmmu.py --checkpoint ${CHECKPOINT} --datasets art_and_design &
  CUDA_VISIBLE_DEVICES=1 python eval/cmmmu/evaluate_cmmmu.py --checkpoint ${CHECKPOINT} --datasets business &
  CUDA_VISIBLE_DEVICES=2 python eval/cmmmu/evaluate_cmmmu.py --checkpoint ${CHECKPOINT} --datasets health_and_medicine &
  CUDA_VISIBLE_DEVICES=3 python eval/cmmmu/evaluate_cmmmu.py --checkpoint ${CHECKPOINT} --datasets humanities_and_social_sciences &
  CUDA_VISIBLE_DEVICES=4 python eval/cmmmu/evaluate_cmmmu.py --checkpoint ${CHECKPOINT} --datasets science &
  CUDA_VISIBLE_DEVICES=5 python eval/cmmmu/evaluate_cmmmu.py --checkpoint ${CHECKPOINT} --datasets technology_and_engineering &
  wait
fi
