# Evaluation

## Directory Structure

```
data
├── flickr30k
│   ├── flickr30k_test_karpathy.json
│   └── Images
├── coco
│   ├── annotations
│   │   ├── coco_karpathy_test_gt.json
│   │   ├── coco_karpathy_test.json
│   │   └── ...
│   ├── train2014
│   ├── val2014
│   └── test2015
├── nocaps
│   ├── nocaps_val_4500_captions.json
│   └── images
├── refcoco
    ├── refcocog_test.jsonl
    ├── refcocog_val.jsonl
    ├── refcoco_testA.jsonl
    ├── refcoco+_testA.jsonl
    ├── refcoco_testB.jsonl
    ├── refcoco+_testB.jsonl
    ├── refcoco_val.jsonl
    └── refcoco+_val.jsonl
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

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-textvqa-val
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
# VizWiz val
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vizwiz-val
# VizWiz test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vizwiz-test
```
For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission).


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

### [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)

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

cd ../..
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
