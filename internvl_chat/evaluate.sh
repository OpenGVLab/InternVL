set -x

CHECKPOINT=${1}
DATASET=${2}
CHECKPOINT="$(pwd)/${CHECKPOINT}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-63665}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}


if  [ ${DATASET} == "mme" ]; then
  cd eval/mme/
  DIRNAME=`basename ${CHECKPOINT}`
  python eval.py --checkpoint ${CHECKPOINT}
  python calculation.py --results_dir ${DIRNAME}
  cd ../../
fi

if  [ ${DATASET} == "caption" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT}
fi

if  [ ${DATASET} == "caption-coco" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127--master_port=${MASTER_PORT}.0.0.1 \
    --nproc_per_node=${GPUS} \
     \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets coco
fi

if  [ ${DATASET} == "caption-flickr30k" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets flickr30k
fi

if  [ ${DATASET} == "caption-nocaps" ]; then
  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets nocaps
fi

if [ ${DATASET} == "vqa" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT}
fi

if [ ${DATASET} == "vqa-okvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets okvqa_val
fi

if [ ${DATASET} == "vqa-textvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets textvqa_val_ocr
fi

if [ ${DATASET} == "vqa-vizwiz-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vizwiz_val
fi

if [ ${DATASET} == "vqa-vizwiz-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vizwiz_test
fi

if [ ${DATASET} == "vqa-vqav2-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_testdev
fi

if [ ${DATASET} == "vqa-ai2d-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ai2diagram_test
fi

if [ ${DATASET} == "vqa-vqav2-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets vqav2_val
fi

if [ ${DATASET} == "vqa-gqa-testdev" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets gqa_testdev_llava
fi

if [ ${DATASET} == "vqa-docvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_val
fi

if [ ${DATASET} == "vqa-docvqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets docvqa_test
fi

if [ ${DATASET} == "vqa-chartqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human,chartqa_test_augmented
fi

if [ ${DATASET} == "vqa-chartqa-test-human" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_human
fi

if [ ${DATASET} == "vqa-chartqa-test-augmented" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets chartqa_test_augmented
fi

if [ ${DATASET} == "vqa-ocrvqa-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ocrvqa_val
fi

if [ ${DATASET} == "vqa-ocrvqa-test" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ocrvqa_test
fi

if [ ${DATASET} == "refcoco" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT}
fi

if [ ${DATASET} == "refcoco-val" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} --datasets refcoco_val
fi

if [ ${DATASET} == "llava-bench" ]; then
    rm -rf results/llava_bench_results_review.jsonl
    python eval/llava_bench/evaluate_llava_bench.py --checkpoint ${CHECKPOINT}
    python -u eval/llava_bench/eval_gpt_review_bench.py \
      --question data/llava-bench-in-the-wild/questions.jsonl \
      --context data/llava-bench-in-the-wild/context.jsonl \
      --rule eval/llava_bench/rule.json \
      --answer-list \
          data/llava-bench-in-the-wild/answers_gpt4.jsonl \
          results/llava_bench_results.jsonl \
      --output \
          results/llava_bench_results_review.jsonl
    python -u eval/llava_bench/summarize_gpt_review.py -f results/llava_bench_results_review.jsonl
fi

if [ ${DATASET} == "pope" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/pope/evaluate_pope.py --checkpoint ${CHECKPOINT} --datasets pope
fi

if [ ${DATASET} == "tiny_lvlm" ]; then
    torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/tiny_lvlm/evaluate_lvlm.py --checkpoint ${CHECKPOINT} --datasets updated_datasets
fi

if [ ${DATASET} == "mmvet" ]; then
    python eval/mmvet/evaluate_mmvet.py --checkpoint ${CHECKPOINT} --datasets mmvet
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

if [ ${DATASET} == "mmbench-dev-en" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_dev_20230712
fi

if [ ${DATASET} == "mmbench-dev-cn" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_dev_cn_20231003
fi

if [ ${DATASET} == "mmbench-test-en" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_test_en_20231003
fi

if [ ${DATASET} == "mmbench-test-cn" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmbench/evaluate_mmbench.py --checkpoint ${CHECKPOINT} --datasets mmbench_test_cn_20231003
fi

if [ ${DATASET} == "scienceqa" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/scienceqa/evaluate_scienceqa.py --checkpoint ${CHECKPOINT} --datasets sqa_test
fi


if [ ${DATASET} == "mmmu-dev" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_dev
fi

if [ ${DATASET} == "mmmu-val" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_validation
fi

if [ ${DATASET} == "mmmu-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT} --datasets MMMU_test
fi


if [ ${DATASET} == "mmvp" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mmvp/evaluate_mmvp.py --checkpoint ${CHECKPOINT} --datasets MMVP
fi


if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_testmini
fi


if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} --datasets MathVista_test
fi

if [ ${DATASET} == "seed" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      eval/seed/evaluate_seed.py --checkpoint ${CHECKPOINT} --datasets SEEDv1
fi
