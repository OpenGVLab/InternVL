# Evaluation

## Directory Structure

```
data
├── flickr30k
│   ├── flickr30k_test_karpathy.json
│   └── Images/
├── coco
│   ├── annotations
│   │   ├── coco_karpathy_test_gt.json
│   │   ├── coco_karpathy_test.json
│   │   └── ...
│   ├── train2014/
│   ├── val2014/
│   └── test2015/
├── nocaps
│   ├── nocaps_val_4500_captions.json
│   └── images/
├── vqav2
│   ├── v2_mscoco_train2014_annotations.json
│   ├── v2_mscoco_train2014_complementary_pairs.json
│   ├── v2_mscoco_val2014_annotations.json
│   ├── v2_OpenEnded_mscoco_test2015_questions.json
│   ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── vqav2_testdev.jsonl
│   ├── vqav2_train.jsonl
│   ├── vqav2_val.jsonl
│   ├── train2014/ -> ../coco/train2014/
│   ├── val2014/ -> ../coco/val2014/
│   └── test2015/ -> ../coco/test2015/
├── okvqa
│   ├── mscoco_train2014_annotations.json
│   ├── mscoco_val2014_annotations.json
│   ├── OpenEnded_mscoco_train2014_questions.json
│   ├── OpenEnded_mscoco_val2014_questions.json
│   ├── okvqa_train.jsonl
│   ├── okvqa_val.jsonl
│   ├── train2014/ -> ../coco/train2014/
│   └── val2014/ -> ../coco/val2014/
├── textvqa
│   ├── textvqa_train_annotations.json
│   ├── textvqa_train.jsonl
│   ├── textvqa_train_questions.json
│   ├── textvqa_val_annotations.json
│   ├── textvqa_val.jsonl
│   ├── textvqa_val_questions.json
│   ├── textvqa_val_llava.jsonl
│   └── train_images/
├── vizwiz
│   ├── vizwiz_test.jsonl
│   ├── vizwiz_train_annotations.json
│   ├── vizwiz_train.jsonl
│   ├── vizwiz_train_questions.json
│   ├── vizwiz_val_annotations.json
│   ├── vizwiz_val.jsonl
│   ├── vizwiz_val_questions.json
│   ├── test/
│   ├── train/
│   ├── val/
│   └── annotations/
├── docvqa
│   ├── test.jsonl
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test/
│   ├── train/
│   └── val/
├── chartqa
│   ├── ChartQA Dataset/
│   │   ├── train/
│   │   ├── test/
│   │   ├── val/
│   ├── test_augmented.jsonl
│   ├── test_human.jsonl
│   ├── train_augmented.jsonl
│   └── train_human.jsonl
├── gqa
│   ├── images/
│   ├── eval.py
│   ├── challenge_all_questions.json
│   ├── challenge_balanced_questions.json
│   ├── llava_gqa_testdev_balanced_qwen_format.jsonl
│   ├── submission_all_questions.json
│   ├── test_all_questions.json
│   ├── test_balanced.jsonl
│   ├── test_balanced_questions.json
│   ├── testdev_all_questions.json
│   ├── testdev_balanced_all_questions.json
│   ├── testdev_balanced_questions.json
│   ├── train_all_questions/
│   ├── train_balanced.jsonl
│   ├── train_balanced_questions.json
│   ├── val_all_questions.json
│   └── val_balanced_questions.json
├── ocrvqa
│   ├── images/
│   ├── ocrvqa_test.jsonl
│   ├── ocrvqa_train.jsonl
│   └── ocrvqa_val.jsonl
├── ai2diagram
│   ├── ai2d/
│   ├── test.jsonl
│   └── train.jsonl
├── scienceqa
│   ├── images/
│   ├── problems.json
│   └── scienceqa_test_img.jsonl
├── refcoco
│   ├── refcocog_test.jsonl
│   ├── refcocog_val.jsonl
│   ├── refcoco_testA.jsonl
│   ├── refcoco+_testA.jsonl
│   ├── refcoco_testB.jsonl
│   ├── refcoco+_testB.jsonl
│   ├── refcoco_val.jsonl
│   └── refcoco+_val.jsonl
├── mme
│   ├── MME_Benchmark_release/
│   └── images/
├── pope
│   ├── coco/
│   │    ├── coco_pope_adversarial.json
│   │    ├── coco_pope_popular.json
│   │    └── coco_pope_random.json
│   ├── val2014/ -> ../coco/val2014/
│   └── llava_pope_test.jsonl
├── tiny_lvlm
│   └── updated_datasets
│       ├── Object_Hallucination
│       ├── ...
│       └── Visual_Reasoning
├── mmbench
│   ├── mmbench_dev_20230712.tsv
│   ├── mmbench_dev_cn_20231003.tsv
│   ├── mmbench_dev_en_20231003.tsv
│   ├── mmbench_test_cn_20231003.tsv
│   └── mmbench_test_en_20231003.tsv
├── llava-bench-in-the-wild
│   ├── answers_gpt4.jsonl
│   ├── ...
│   └── images/
```

## Image Caption

### [COCO karpathy test](https://cocodataset.org/)

> COCO images are used in VQAv2/OK-VQA/RefCOCO/RefCOCO+/RefCOCOg. Make sure you have already downloaded COCO images before evaluating on these benchmarks.

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/coco && cd data/coco

# download coco images
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip
wget http://images.cocodataset.org/zips/test2015.zip && unzip test2015.zip

mkdir -p annotations && cd annotations/
# download converted annotation files
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test.json
wget https://github.com/OpenGVLab/InternVL/releases/download/data/coco_karpathy_test_gt.json

cd ../../../
```

</details>

<details>
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> caption-coco
```

</details>

### [Flickr30K karpathy test](https://bryanplummer.com/Flickr30kEntities/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/flickr30k && cd data/flickr30k

# download images from https://bryanplummer.com/Flickr30kEntities/
# karpathy split annotations can be downloaded from https://cs.stanford.edu/people/karpathy/deepimagesent/
# download converted files
wget https://github.com/OpenGVLab/InternVL/releases/download/data/flickr30k_test_karpathy.json

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> caption-flickr30k
```

</details>

### [NoCaps val](https://nocaps.org/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/nocaps && cd data/nocaps

# download images from https://nocaps.org/download
# original annotations can be downloaded from https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
wget https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> caption-nocaps
```

</details>

## General VQA

### [VQAv2 val & test-dev](https://visualqa.org/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/vqav2 && cd data/vqav2

# make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./
ln -s ../coco/test2015 ./

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

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
# VQAv2-val
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vqav2-val
# VQAv2-testdev
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vqav2-testdev
```

For the testdev set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission).

</details>

### [OKVQA val](https://okvqa.allenai.org/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/okvqa && cd data/okvqa

# make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./

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
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-okvqa-val
```

</details>

### [TextVQA val](https://textvqa.org/)

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
wget https://github.com/OpenGVLab/InternVL/releases/download/data/textvqa_val_llava.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-textvqa-val
```

</details>

### [VizWiz val & test](https://vizwiz.org/tasks-and-datasets/vqa/)

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
# VizWiz val
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vizwiz-val
# VizWiz test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vizwiz-test
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission).

</details>

### [DocVQA val & test](https://www.docvqa.org/datasets)

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
# DocVQA-val
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-docvqa-val
# DocVQA-test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-docvqa-test
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

</details>

### [ChartQA test-human & test-augmented](https://aclanthology.org/2022.findings-acl.177/)

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
<summary>Evaluation</summary>

```bash
# ChartQA-test-human
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-chartqa-test-human
# ChartQA-test-augmented
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-chartqa-test-augmented
```

</details>

### [GQA testdev](https://cs.stanford.edu/people/dorarad/gqa/about.html)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/gqa && cd data/gqa

# download images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/testdev_balanced.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/train_balanced.jsonl
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_gqa_testdev_balanced_qwen_format.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-gqa-testdev
```

</details>

### [OCRVQA val & test](https://ocr-vqa.github.io/)

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
<summary>Evaluation</summary>

```bash
# OCRVQA-val
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-ocrvqa-val
# OCRVQA-test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-ocrvqa-test
```

</details>

### [AI2Diagram test](https://allenai.org/data/diagrams)

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
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-ai2d-test
```

</details>

### [ScienceQA test](https://github.com/lupantech/ScienceQA)

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
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> scienceqa-test
```

</details>

## Refer Expression Comprehension

### RefCOCO/RefCOCO+/RefCOCO-g

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/refcoco && cd data/refcoco

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testB.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testB.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_test.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> refcoco
```

</details>

## MultiModal Dialogue

### [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/mme && cd data/mme

# 1. Download MME images and eval_tool from the [MME repo](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/README.md)
# 2. Rearrange images by executing `python get_images.py`
python get_images.py
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
# single GPU testing
CUDA_VISIBLE_DEVICES=0 sh evaluate.sh <checkpoint> mme
```

</details>

### [MMBench dev & test](https://github.com/open-compass/mmbench/?tab=readme-ov-file)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/mmbench && cd data/mmbench

# download csv files of mmbench
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_en_20231003.tsv

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
# mmbench_dev_20230712
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> mmbench-dev-en
# mmbench_dev_cn_20231003
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> mmbench-dev-cn
# mmbench_test_en_20231003
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> mmbench-test-en
# mmbench_test_cn_20231003
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> mmbench-test-cn
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission).

</details>

### [POPE](https://github.com/AoiDragon/POPE/tree/main)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/pope && cd data/pope

# make sure you have downloaded COCO images
ln -s ../coco/val2014 ./
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_pope_test.jsonl

# download `coco` from POPE
mkdir -p coco && cd coco
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json
cd ../../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> pope
```

</details>

### MMMU

### CMMMU

### [Tiny LVLM](https://github.com/OpenGVLab/Multi-Modality-Arena/tree/main/tiny_lvlm_evaluation)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/tiny_lvlm && cd data/tiny_lvlm

# download dataset from https://github.com/OpenGVLab/Multi-Modality-Arena/tree/main/tiny_lvlm_evaluation
# i.e., download `updated_datasets.tar.gz` from https://drive.google.com/file/d/1PuFC612XzOmKwzRldtBb1CFZnIjiR7we/view
tar -xzvf updated_datasets.tar.gz

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> tiny_lvlm
```

</details>

### [LLaVA Bench](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild)

<details>
<summary>Data Preparation</summary>

```bash
cd data/
# download dataset from https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild
git clone https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild
cd llava-bench-in-the-wild/
rm -rf images && mkdir -p images && cd images
# download all 24 images
wget https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/001.jpg
# ...
wget https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/024.jpg
cd ../../../
```

</details>

<details>
<summary>Evaluation</summary>

```bash
# single GPU testing
export OPENAI_API_KEY='your_gpt4_key'
CUDA_VISIBLE_DEVICES=0 sh evaluate.sh <checkpoint> llava-bench
```

</details>
