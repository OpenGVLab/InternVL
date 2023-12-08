# Evaluation

## Dependencies

```bash
pip install pycocoevalcap tqdm
```

## Image Caption

### [Flickr30K](https://bryanplummer.com/Flickr30kEntities/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/flickr && cd data/flickr

# download images from https://bryanplummer.com/Flickr30kEntities/

# karpathy split annotations can be downloaded from https://cs.stanford.edu/people/karpathy/deepimagesent/

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/flickr30k/flickr30k_karpathy_test.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/flickr30k/flickr30k_karpathy_train.json

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
ds="flickr"
checkpoint=/PATH/TO/CHECKPOINT
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_caption.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```

</details>

### [Nocaps](https://nocaps.org/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/nocaps && cd data/nocaps

# download images from https://nocaps.org/download

# original annotations can be downloaded from https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/nocaps/nocaps_val.json

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
ds="nocaps"
checkpoint=/PATH/TO/CHECKPOINT
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_caption.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```

</details>

## [COCO](https://cocodataset.org/)

> COCO images are used in VQAv2/OK-VQA/RefCOCO/RefCOCO+/RefCOCOg, make sure you have already downloaded COCO images before evaluate on these benchmarks.

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/coco && cd data/coco

# download coco2014 images
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip && unzip test2015.zip

cd ../..
```

</details>

## General VQA

### [VQAv2](https://visualqa.org/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/vqav2 && cd data/vqav2

# make sure you have downloaded COCO images

# download questions and annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && unzip v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && unzip v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && unzip v2_Questions_Test_mscoco.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_testdev.jsonl
```

</details>

<details>
<summary>Evaluate</summary>

```bash
checkpoint=/PATH/TO/CHECKPOINT
for ds in "vqav2_val" "vqav2_testdev"
    python -m torch.distributed.launch --use-env \
        --nproc_per_node ${NPROC_PER_NODE:-8} \
        --nnodes ${WORLD_SIZE:-1} \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        evaluate_vqa.py \
        --checkpoint $checkpoint \
        --dataset $ds \
        --batch-size 8 \
        --num-workers 2
```

</details>

### [OKVQA](https://okvqa.allenai.org/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/okvqa && cd data/okvqa

# download annotations and questions
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip && unzip mscoco_train2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip && unzip OpenEnded_mscoco_train2014_questions.json.zip
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip && unzip mscoco_val2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip && unzip OpenEnded_mscoco_val2014_questions.json.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/okvqa/okvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/okvqa/okvqa_val.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
ds="okvqa_val"
checkpoint=/PATH/TO/CHECKPOINT
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```

</details>

### [TextVQA](https://textvqa.org/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/textvqa && cd data/textvqa

# download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && unzip train_val_images.zip

# download annotations and questions
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
ds="textvqa_val"
checkpoint=/PATH/TO/CHECKPOINT
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```

</details>

### [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/vizwiz && cd data/vizwiz

# download images
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip && unzip train.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip && unzip val.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip && unzip test.zip

# download annotations
wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip && unzip Annotations.zip

# download converted files
# train
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train.jsonl
# val
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val.jsonl
# test
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_test.jsonl
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
# evaluate vqa score on vizwiz val split
ds="vizwiz_val"
checkpoint=/PATH/TO/CHECKPOINT
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2

```

</details>

### [DocVQA](https://www.docvqa.org/datasets)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/docvqa && cd data/docvqa

# download images and annotations from https://www.docvqa.org/datasets

# download converted files
# train
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/train.jsonl
# val
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/val.jsonl
# test
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/test.jsonl
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
# evaluate vqa score on docvqa val split
ds="docvqa_val"
checkpoint=/PATH/TO/CHECKPOINT
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```

</details>

### [ChartQA](https://aclanthology.org/2022.findings-acl.177/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/chartqa && cd data/chartqa

# download images from https://drive.google.com/file/d/1Lm_w6zeET1Hyl_9ks6w5nEsgpoyPHalV/view

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_augmented.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_augmented.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
checkpoint=/PATH/TO/CHECKPOINT
for ds in "chartqa_test_human" "chartqa_test_augmented"
    python -m torch.distributed.launch --use-env \
        --nproc_per_node ${NPROC_PER_NODE:-8} \
        --nnodes ${WORLD_SIZE:-1} \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        evaluate_vqa.py \
        --checkpoint $checkpoint \
        --dataset $ds \
        --batch-size 8 \
        --num-workers 2
```

</details>

### [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/gqa && cd data/gqa

# download images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip

# download converted files
https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/test_balanced.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/train_balanced.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
checkpoint=/PATH/TO/CHECKPOINT
ds="gqa_testdev"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```

</details>

### [OCRVQA](https://ocr-vqa.github.io/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/ocrvqa && cd data/ocrvqa

# download images by following instructions at https://ocr-vqa.github.io/kvqa_ProjectFiles/README.txt

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ocrvqa/ocrvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ocrvqa/ocrvqa_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ocrvqa/ocrvqa_test.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
checkpoint=/PATH/TO/CHECKPOINT
ds="ocrvqa_test"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```

</details>

### [AI2Diagram](https://allenai.org/data/diagrams)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/ai2diagram && cd data/ai2diagram

# download images
wget https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ai2diagram/train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/ai2diagram/test.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
checkpoint=/PATH/TO/CHECKPOINT
ds="ai2diagram_test"
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_vqa.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```

</details>

### [ScienceQA](https://github.com/lupantech/ScienceQA)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/scienceqa/images && cd data/scienceqa/images

# download images
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip && unzip test.zip

cd ..

# download original questions
wget https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/problems.json

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/scienceqa/scienceqa_test_img.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
ds="scienceqa_test_img"
checkpoint=/PATH/TO/CHECKPOINT
python -m torch.distributed.launch --use-env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    evaluate_multiple_choice.py \
    --checkpoint $checkpoint \
    --dataset $ds \
    --batch-size 8 \
    --num-workers 2
```

</details>

## Refer Expression Comprehension

### RefCOCO

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/refcoco && cd data/refcoco

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testB.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
checkpoint=/PATH/TO/CHECKPOINT
for ds in "refcoco_val" "refcoco_testA" "refcoco_testB"
    python -m torch.distributed.launch --use-env \
        --nproc_per_node ${NPROC_PER_NODE:-8} \
        --nnodes ${WORLD_SIZE:-1} \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        evaluate_grounding.py \
        --checkpoint $checkpoint \
        --dataset $ds \
        --batch-size 8 \
        --num-workers 2
```

</details>

### RefCOCO+

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/refcoco+ && cd data/refcoco+

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testB.jsonl

cd ../..
```

</details>

<details>
<summary>Data Preparation</summary>

```bash
checkpoint=/PATH/TO/CHECKPOINT
for ds in "refcoco+_val" "refcoco+_testA" "refcoco+_testB"
    python -m torch.distributed.launch --use-env \
        --nproc_per_node ${NPROC_PER_NODE:-8} \
        --nnodes ${WORLD_SIZE:-1} \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        evaluate_grounding.py \
        --checkpoint $checkpoint \
        --dataset $ds \
        --batch-size 8 \
        --num-workers 2
```

</details>

### RefCOCOg

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/refcocog && data/refcocog

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_test.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluate</summary>

```bash
checkpoint=/PATH/TO/CHECKPOINT
for ds in "refcocog_val" "refcocog_test"
    python -m torch.distributed.launch --use-env \
        --nproc_per_node ${NPROC_PER_NODE:-8} \
        --nnodes ${WORLD_SIZE:-1} \
        --node_rank ${RANK:-0} \
        --master_addr ${MASTER_ADDR:-127.0.0.1} \
        --master_port ${MASTER_PORT:-12345} \
        evaluate_grounding.py \
        --checkpoint $checkpoint \
        --dataset $ds \
        --batch-size 8 \
        --num-workers 2
```

</details>
