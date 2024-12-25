# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for visual question answering across 9 datasets: `VQAv2`, `OKVQA`, `TextVQA`, `Vizwiz`, `DocVQA`, `ChartQA`, `AI2D`, `InfoVQA`, and `GQA`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `InternVL/internvl_chat/data` folder.

### VQAv2

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/vqav2 && cd data/vqav2

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./
ln -s ../coco/test2015 ./

# Step 3: Download questions and annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && unzip v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && unzip v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && unzip v2_Questions_Test_mscoco.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_testdev.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/vqav2
├── train2014 -> ../coco/train2014
├── val2014 -> ../coco/val2014
├── test2015 -> ../coco/test2015
├── v2_mscoco_train2014_annotations.json
├── v2_mscoco_train2014_complementary_pairs.json
├── v2_mscoco_val2014_annotations.json
├── v2_OpenEnded_mscoco_test2015_questions.json
├── v2_OpenEnded_mscoco_test-dev2015_questions.json
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── vqav2_testdev.jsonl
├── vqav2_train.jsonl
└── vqav2_val.jsonl
```

### OKVQA

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/okvqa && cd data/okvqa

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./

# Step 3: Download annotations and questions
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip && unzip mscoco_train2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip && unzip OpenEnded_mscoco_train2014_questions.json.zip
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip && unzip mscoco_val2014_annotations.json.zip
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip && unzip OpenEnded_mscoco_val2014_questions.json.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/okvqa/okvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/okvqa/okvqa_val.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/okvqa
├── mscoco_train2014_annotations.json
├── mscoco_val2014_annotations.json
├── okvqa_train.jsonl
├── okvqa_val.jsonl
├── OpenEnded_mscoco_train2014_questions.json
├── OpenEnded_mscoco_val2014_questions.json
├── test2014 -> ../coco/test2014
└── val2014 -> ../coco/val2014
```

### TextVQA

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/textvqa && cd data/textvqa

# Step 2: Download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && unzip train_val_images.zip

# Step 3: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_questions.json
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val_llava.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/textvqa
├── TextVQA_Rosetta_OCR_v0.2_test.json
├── TextVQA_Rosetta_OCR_v0.2_train.json
├── TextVQA_Rosetta_OCR_v0.2_val.json
├── textvqa_train_annotations.json
├── textvqa_train.jsonl
├── textvqa_train_questions.json
├── textvqa_val_annotations.json
├── textvqa_val.jsonl
├── textvqa_val_llava.jsonl
├── textvqa_val_questions.json
└── train_images
```

### VizWiz

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/vizwiz && cd data/vizwiz

# Step 2: Download images
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip && unzip train.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip && unzip val.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip && unzip test.zip

# Step 3: Download annotations
wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip && unzip Annotations.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vizwiz/vizwiz_test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/vizwiz
├── annotations
├── test
├── train
├── val
├── vizwiz_test.jsonl
├── vizwiz_train_annotations.json
├── vizwiz_train.jsonl
├── vizwiz_train_questions.json
├── vizwiz_val_annotations.json
├── vizwiz_val.jsonl
└── vizwiz_val_questions.json
```

### DocVQA

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/docvqa && cd data/docvqa

# Step 2: Download images and annotations
wget https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz --no-check-certificate # (optional)
wget https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz --no-check-certificate
wget https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz --no-check-certificate

# Step 3: Unzip files
tar -zxvf train.tar.gz
tar -zxvf val.tar.gz
tar -zxvf test.tar.gz

# Step 4: Download converted jsonl files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/docvqa
├── test
├── test.jsonl
├── train
├── train.jsonl
├── val
└── val.jsonl
```

### AI2D

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/ai2diagram && cd data/ai2diagram

# Step 2: Download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/ai2d_test_vlmevalkit.jsonl -O test_vlmevalkit.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/AI2D_TEST.zip && unzip AI2D_TEST.zip

# Step 3: Download images from Google Drive (optional, provided by InternLM-XComposer)
# https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing
# images should be placed in `data/ai2diagram/ai2d/abc_images` and `data/ai2diagram/ai2d/images`
cd ../..
```

After preparation is complete, the directory structure is:

```
data/ai2diagram
 ├── test_vlmevalkit.jsonl
 ├── ai2d # (optional)
 │    ├── abc_images
 │    └── images
 └── AI2D_TEST
```

### InfoVQA

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/infographicsvqa && cd data/infographicsvqa

# Step 2: Download images and annotations from https://rrc.cvc.uab.es/?ch=17&com=downloads
# infographicsVQA_test_v1.0.json, infographicsVQA_val_v1.0_withQT.json, infographicVQA_train_v1.0.json

# Step 3: Download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_val.jsonl -O val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_test.jsonl -O test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/infographicsvqa
├── infographicsvqa_images
├── infographicsVQA_test_v1.0.json
├── infographicsVQA_val_v1.0_withQT.json
├── infographicVQA_train_v1.0.json
├── test.jsonl
└── val.jsonl
```

### ChartQA

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/chartqa && cd data/chartqa

# Step 2: download images from
# https://drive.google.com/file/d/1Lm_w6zeET1Hyl_9ks6w5nEsgpoyPHalV/view

# Step 3: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_augmented.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_human.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_augmented.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/chartqa
 ├── ChartQA Dataset
 │    ├── test
 │    ├── train
 │    └── val
 ├── test_augmented.jsonl
 ├── test_human.jsonl
 ├── train_augmented.jsonl
 └── train_human.jsonl
```

### GQA

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/gqa && cd data/gqa

# Step 2: Download the official evaluation script
wget https://nlp.stanford.edu/data/gqa/eval.zip
unzip eval.zip

# Step 3: Download images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/testdev_balanced.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/gqa/train_balanced.jsonl
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_gqa_testdev_balanced_qwen_format.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/gqa
├── challenge_all_questions.json
├── challenge_balanced_questions.json
├── eval.py
├── images
├── llava_gqa_testdev_balanced_qwen_format.jsonl
├── readme.txt
├── submission_all_questions.json
├── test_all_questions.json
├── test_balanced.jsonl
├── test_balanced_questions.json
├── testdev_all_questions.json
├── testdev_balanced_all_questions.json
├── testdev_balanced_predictions.json
├── testdev_balanced_questions.json
├── train_all_questions
├── train_balanced.jsonl
├── train_balanced_questions.json
├── val_all_questions.json
└── val_balanced_questions.json
```

## 🏃 Evaluation Execution

> ⚠️ Note: For testing InternVL (1.5, 2.0, 2.5, and later versions), always enable `--dynamic` to perform dynamic resolution testing.

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
torchrun --nproc_per_node=8 eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS} --dynamic
```

Alternatively, you can run the following simplified command:

```shell
# Test VQAv2 val
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-vqav2-val --dynamic
# Test VQAv2 testdev
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-vqav2-testdev --dynamic
# Test OKVQA val
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-okvqa-val --dynamic
# Test Vizwiz val
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-vizwiz-val --dynamic
# Test Vizwiz test
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-vizwiz-test --dynamic
# Test GQA testdev
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-gqa-testdev --dynamic
# Test AI2D test
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-ai2d-test --dynamic
# Test TextVQA val
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-textvqa-val --dynamic
# Test ChartQA test-human & test-augmented
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-chartqa-test --dynamic --max-num 12
# Test DocVQA val
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-docvqa-val --dynamic --max-num 18
# Test DocVQA test
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-docvqa-test --dynamic --max-num 18
# Test InfoVQA val
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-infovqa-val --dynamic --max-num 24
# Test InfoVQA test
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-infovqa-test --dynamic --max-num 24
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default     | Description                                                                                                       |
| ---------------- | ------ | ----------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`        | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `okvqa_val` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`     | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`         | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`     | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`     | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
