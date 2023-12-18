set -x

CHECKPOINT=${1}
DATASET=${2}
CHECKPOINT="/mnt/petrelfs/wangwenhai/workspace/InternVL-release/internvl_chat/${CHECKPOINT}"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "CHECKPOINT: ${CHECKPOINT}"

if  [ ${DATASET} == "mme" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  python eval.py --template "vicuna_v1.1" --model_path ${CHECKPOINT}
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
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1'
fi

if  [ ${DATASET} == "caption-coco" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets coco
fi

if  [ ${DATASET} == "caption-flickr30k" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets flickr30k
fi

if  [ ${DATASET} == "caption-nocaps" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets nocaps
fi

if [ ${DATASET} == "vqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1'
fi

if [ ${DATASET} == "vqa-okvqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets okvqa_val
fi

if [ ${DATASET} == "vqa-textvqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets textvqa_val_ocr
fi

if [ ${DATASET} == "vqa-vizwiz" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets vizwiz_val
fi

if [ ${DATASET} == "vqa-vizwiz-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets vizwiz_test
fi

if [ ${DATASET} == "vqa-vqav2-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets vqav2_testdev
fi

if [ ${DATASET} == "vqa-ai2d" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets ai2diagram_test
fi

if [ ${DATASET} == "vqa-vqav2" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets vqav2_val
fi

if [ ${DATASET} == "vqa-gqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1' --datasets gqa_testdev_llava
fi

if [ ${DATASET} == "refcoco" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=63667 \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1'
fi

if [ ${DATASET} == "llava-bench" ]; then
    python eval/llava_bench/evaluate_vqa.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1'
fi

if [ ${DATASET} == "pope" ]; then
    python eval/llava_bench/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets pope --template 'vicuna_v1.1'
fi

if [ ${DATASET} == "mmvet" ]; then
    python eval/llava_bench/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets mmvet --template 'vicuna_v1.1'
fi

if [ ${DATASET} == "mmbench" ]; then
    cd eval/mmbench/
    python eval.py --checkpoint ${CHECKPOINT} --template 'vicuna_v1.1'
fi
