# InternVL-Chat

This folder contains the implementation of the InternVL-Chat.

## 🛠️ Installation

See [INSTALLATION.md](../INSTALLATION.md)

In addition, using this codebase requires executing the following steps:

- Install other requirements:

  ```bash
  pip install --upgrade pip  # enable PEP 660 support
  pip install -e .
  ```

## 📦 Model Preparation

| model name              | type | download                                                               |  size   |
| ----------------------- | ---- | ---------------------------------------------------------------------- | :-----: |
| InternViT-300M-448px    | ViT  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-300M-448px)    | 0.6 GB  |
| InternViT-6B-448px-V1-2 | ViT  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2) | 11.1 GB |
| InternViT-6B-448px-V1-5 | ViT  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) | 11.1 GB |
| Nous-Hermes-2-Yi-34B    | LLM  | 🤗 [HF link](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) | 65.0 GB |

Please download the above model weights and place them in the `pretrained/` folder.

```sh
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternViT-300M-448px --local-dir InternViT-300M-448px
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternViT-6B-448px-V1-2 --local-dir InternViT-6B-448px-V1-2
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternViT-6B-448px-V1-5 --local-dir InternViT-6B-448px-V1-5
huggingface-cli download --resume-download --local-dir-use-symlinks False NousResearch/Nous-Hermes-2-Yi-34B --local-dir Nous-Hermes-2-Yi-34B
```

The directory structure is:

```sh
pretrained
│── InternViT-300M-448px/
│── InternViT-6B-448px-V1-2/
│── InternViT-6B-448px-V1-5/
└── Nous-Hermes-2-Yi-34B/
```

## 🔥 Supervised Fine-tuning

### Prepare Training Datasets

Inspired by LLaVA-NeXT, we adopted a data-efficient SFT strategy to train InternVL-Chat-V1-2, utilizing approximately 1.2M of visual instruction tuning samples in total, all of which are fully open-source. In a macro sense, we build upon [ShareGPT-4V](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md#prepare-images) and additionally integrate [LLaVA-ZH](https://huggingface.co/datasets/openbmb/llava_zh), [DVQA](https://github.com/kushalkafle/DVQA_dataset), [ChartQA](https://github.com/vis-nlp/ChartQA), [AI2D](https://allenai.org/data/diagrams), [DocVQA](https://www.docvqa.org/datasets), [GeoQA+](https://github.com/SCNU203/GeoQA-Plus), and [SynthDoG-EN](https://huggingface.co/datasets/naver-clova-ix/synthdog-en). Most of the data remains consistent with LLaVA-NeXT.

First, download the [annotation files](https://huggingface.co/OpenGVLab/InternVL/resolve/main/playground.zip) and place them in the `playground/opensource/` folder.

Second, download all the images we used.

- AI2D: [ai2d_images](https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing) (provided by InternLM-XComposer)
- ChartQA: [ChartQA Dataset](https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- DocVQA: [train](https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz), [val](https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz), [test](https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz)
- DVQA: [images](https://drive.google.com/file/d/1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ/view)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- LLaVA-Pretrain: [images](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing). We save all files as `.jpg`
- SAM: We only use 000000~000050.tar for now. You can quickly download 9K images from [here](https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link).
- TextVQA: [trainvalimages](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- SynthDoG-EN: We only use 00000~00004 parquet files for now, with a total of 30K images. We provide the converted [images](https://huggingface.co/OpenGVLab/InternVL/resolve/main/synthdog-en-images.zip).
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- WebData: [images](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing). Only for academic usage.
- GeoQA+: [GeoQA+](https://drive.google.com/file/d/1KL4_wIzr3p8XSKMkkLgYcYwCbb0TzZ9O/view) [images](https://huggingface.co/OpenGVLab/InternVL/resolve/main/geoqa%2B_images.zip)

Then, organize the data as follows in `playground/data`:

```none
playground/
├── opensource
│   ├── ai2d_train_12k.jsonl
│   ├── chartqa_train_18k.jsonl
│   ├── docvqa_train_10k.jsonl
│   ├── dvqa_train_200k.jsonl
│   ├── geoqa+.jsonl
│   ├── llava_instruct_150k_zh.jsonl
│   ├── sharegpt4v_instruct_gpt4-vision_cap100k.jsonl
│   ├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl
│   └── synthdog_en.jsonl
├── data
│   ├── ai2d
│   │   ├── abc_images
│   │   └── images
│   ├── chartqa
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── coco
│   │   └── train2017
│   ├── docvqa
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── dvqa
│   │   └── images
│   ├── gqa
│   │   └── images
│   ├── llava
│   │   └── llava_pretrain
│   │       └── images
│   ├── ocr_vqa
│   │   └── images
│   ├── sam
│   │   └── images
│   ├── share_textvqa
│   │   └── images
│   ├── synthdog-en
│   │   └── images
│   ├── textvqa
│   │   └── train_images
│   ├── vg
│   │   ├── VG_100K
│   │   └── VG_100K_2
│   ├── web-celebrity
│   │   └── images
│   ├── web-landmark
│   │   └── images
│   ├── wikiart
│   │   └── images
│   ├── geoqa+
│   │   └── images
```

### Start Training

We provide slurm scripts for multi-node multi-GPU training. You can use either 32 or 64 GPUs to train this model. If you use 64 GPUs, training will take approximately 18 hours.

- If you encounter an OOM error, you can decrease the `PER_DEVICE_BATCH_SIZE`, for example, set `PER_DEVICE_BATCH_SIZE=4`.

```sh
# using 32 GPUs
PARTITION='your partition' GPUS=32 PER_DEVICE_BATCH_SIZE=8 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune.sh
# using 64 GPUs
PARTITION='your partition' GPUS=64 PER_DEVICE_BATCH_SIZE=8 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune.sh
```

The hyperparameters used for fine-tuning are listed in the following table. And, you can view the training logs in tensorboard at [here](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2/tensorboard).

| Hyperparameter     | Trainable Param  | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| ------------------ | ---------------- | ----------------- | ------------- | ------ | ---------- | ------------ |
| InternVL-Chat-V1-2 | 40B (full model) | 512               | 1e-5          | 1      | 2048       | 0.05         |

### Continued Fine-tune

See [this document](../document/how_to_finetune_internvl_chat_v1_2_on_a_custom_dataset.md) to finetune InternVL-Chat-V1-2.

## 📊 Evaluation

**OCR-related Benchmarks**

Note: TextVQA contains two scores, representing not using or using Rosetta OCR tokens, respectively.

| model                                                                               | #param | DocVQA<br>(val/test) | ChartVQA<br>(avg. test) | InfoVQA<br>(val/test) | TextVQA<br>(val, wo/w OCR) | OCRBench | AI2D |
| ----------------------------------------------------------------------------------- | ------ | -------------------- | ----------------------- | --------------------- | -------------------------- | -------- | ---- |
| [InternVL&#8209;Chat&#8209;V1&#8209;1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1)           | 19B    | 47.6 / 48.1          | 59.9                    | 33.3 / 32.0           | 64.2 / 68.6                | 530      | 72.4 |
| [InternVL&#8209;Chat&#8209;V1&#8209;2](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2)           | 40B    | 56.4 / 57.7          | 68.0                    | 36.0 / 39.5           | 67.5 / 72.5                | 569      | 79.0 |
| [InternVL&#8209;Chat&#8209;V1&#8209;2&#8209;Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) | 40B    | 56.9 / 56.8          | 72.8                    | 40.9 / 40.6           | 71.2 / 74.1                | 598      | 78.9 |
| [InternVL&#8209;Chat&#8209;V1&#8209;5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)           | 26B    | 90.5 / 90.8          | 83.8                    | 72.4 / 72.5           | 80.6 / -                   | 724      | 80.7 |

**MultiModal Benchmark**

| model                                                                               | #param | MME            | MMB<br>(dev/test) | MMB&#8209;CN<br>(dev/test) | CCBench | MMVet | MMMU<br>(val/test)                                                                 | MathVista<br>(testmini) | Hallusion<br>Bench | RealWorld<br/>QA | SEEDv1<br>(image) | CMMMU<br>(val/test) | POPE | MMVP | Tiny LVLM | LLaVA Wild |
| ----------------------------------------------------------------------------------- | ------ | -------------- | ----------------- | -------------------- | ------- | ----- | ---------------------------------------------------------------------------------- | ----------------------- | ------------------ | ---------------- | ----------------- | ------------------- | ---- | ---- | --------- | ---------- |
| [InternVL&#8209;Chat&#8209;V1&#8209;1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1)           | 19B    | 1659.8 / 361.4 | 76.7 / 75.4       | 71.9 / 70.3          | 43.3    | 46.7  | 39.1 / 35.3                                                                        | 34.5                    | 36.1               | 58.0             | 73.2              | 34.8 / 34.0         | 87.1 | 44.7 | 343.2     | 73.2       |
| [InternVL&#8209;Chat&#8209;V1&#8209;2](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2)           | 40B    | 1686.8 / 488.6 | 81.4 / 82.2       | 79.5 / 81.2          | 58.6    | 48.9  | 51.6 / [46.2](https://eval.ai/web/challenges/challenge-page/2179/leaderboard/5377) | 47.7                    | 47.6               | 67.5             | 75.6              | -                   | 88.0 | 56.7 | 350.3     | 85.0       |
| [InternVL&#8209;Chat&#8209;V1&#8209;2&#8209;Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) | 40B    | 1625.2 / 552.9 | 83.4 / 83.8       | 81.6 / 82.0          | 55.9    | 47.9  | 50.3 / 45.6                                                                        | 59.9                    | 47.4               | 67.8             | 76.4              | -                   | 88.7 | 58.7 | 353.9     | 84.6       |
| [InternVL&#8209;Chat&#8209;V1&#8209;5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)           | 26B    | 1637.8 / 550.0 | - / 82.2          | - / 82.0             | 70.0    | 62.8  | 45.2 / -                                                                           | 53.5                    | 49.3               | 66.0             | 76.0              | -                   | 88.3 | 57.3 | 356.8     | 94.7       |

**Visual Question Answering & Image Captioning**

| model                                                                               | #param | OKVQA<br>(val) | VizWiz<br>(val/test) | GQA<br>(test) | SQA<br>(image) | VQAv2<br>(testdev) | COCO<br>(test) | Flickr30K<br>(test) | NoCaps<br>(val) |
| ----------------------------------------------------------------------------------- | ------ | -------------- | -------------------- | ------------- | -------------- | ------------------ | -------------- | ------------------- | --------------- |
| [InternVL&#8209;Chat&#8209;V1&#8209;1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1)           | 19B    | 64.1           | 59.0 / 57.3          | 62.5          | 90.1           | 80.9               | 142.2          | 84.8                | 120.8           |
| [InternVL&#8209;Chat&#8209;V1&#8209;2](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2)           | 40B    | 62.5           | 61.9 / 60.0          | 64.0          | 83.3           | -                  | 113.9          | 92.9                | 112.5           |
| [InternVL&#8209;Chat&#8209;V1&#8209;2&#8209;Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) | 40B    | 67.6           | 61.3 / 59.5          | 66.9          | 98.1           | -                  | 143.4          | 89.5                | 125.8           |
| [InternVL&#8209;Chat&#8209;V1&#8209;5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)           | 26B    | 62.0           | 63.5 / -             | 65.7          | 94.0           | -                  | 98.4           | 81.2                | 99.6            |

**Visual Grounding**

| model                                                                               | #param | RefCOCO<br>(val) | RefCOCO<br>(testA) | RefCOCO<br>(testB) | RefCOCO+<br>(val) | RefCOCO+<br>(testA) | RefCOCO+<br>(testB) | RefCOCO&#8209;g<br>(val) | RefCOCO&#8209;g<br>(test) |
| ----------------------------------------------------------------------------------- | ------ | ---------------- | ------------------ | ------------------ | ----------------- | ------------------- | ------------------- | ------------------ | ------------------- |
| [InternVL&#8209;Chat&#8209;V1&#8209;1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1)           | 19B    | 84.7             | 89.9               | 78.6               | 78.5              | 85.6                | 70.1                | 81.0               | 81.4                |
| [InternVL&#8209;Chat&#8209;V1&#8209;2](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2)           | 40B    | 74.4             | 80.3               | 66.5               | 70.7              | 77.6                | 62.0                | 69.2               | 70.0                |
| [InternVL&#8209;Chat&#8209;V1&#8209;2&#8209;Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) | 40B    | 90.2             | 93.4               | 85.5               | 85.3              | 90.4                | 79.7                | 88.5               | 88.8                |
| [InternVL&#8209;Chat&#8209;V1&#8209;5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)           | 26B    | 91.4             | 93.7               | 87.1               | 87.0              | 92.3                | 80.9                | 88.5               | 89.3                |

## 📊 Evaluation (Legacy Models)

| model         | QLLaMA | LLM          | res | COCO  | Flickr | NoCaps | VQAv2 | GQA  | VizWiz | TextVQA | MME    | POPE | Download |
| ------------- | ------ | ------------ | --- | ----- | ------ | ------ | ----- | ---- | ------ | ------- | ------ | ---- | -------- |
| InternVL&#8209;Chat | ✔️     | frozen V&#8209;7B  | 224 | 141.4 | 89.7   | 120.5  | 72.3  | 57.7 | 44.5   | 42.1    | 1298.5 | 85.2 | TODO     |
| InternVL&#8209;Chat | ✔️     | frozen V&#8209;13B | 224 | 142.4 | 89.9   | 123.1  | 71.7  | 59.5 | 54.0   | 49.1    | 1317.2 | 85.4 | TODO     |
| InternVL&#8209;Chat | ✔️     | V&#8209;13B        | 336 | 146.2 | 92.2   | 126.2  | 81.2  | 66.6 | 58.5   | 61.5    | 1586.4 | 87.6 | TODO     |

## ❓ How to Evaluate

### Image Caption Benchmarks

#### [COCO Karpathy test](https://cocodataset.org/)

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
GPUS=8 sh evaluate.sh <checkpoint> caption-coco [--dynamic]
```

</details>

#### [Flickr30K Karpathy test](https://bryanplummer.com/Flickr30kEntities/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/flickr30k && cd data/flickr30k

# download images from https://bryanplummer.com/Flickr30kEntities/
# karpathy split annotations can be downloaded from the following link:
# https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/flickr30k_test_karpathy.txt
# this file is provided by the clip-benchmark repository.
# We convert this txt file to json format, download the converted file:
wget https://github.com/OpenGVLab/InternVL/releases/download/data/flickr30k_test_karpathy.json

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> caption-flickr30k [--dynamic]
```

</details>

#### [NoCaps val](https://nocaps.org/)

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
GPUS=8 sh evaluate.sh <checkpoint> caption-nocaps [--dynamic]
```

</details>

### General VQA Benchmarks

#### [VQAv2 val & test-dev](https://visualqa.org/)

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
GPUS=8 sh evaluate.sh <checkpoint> vqa-vqav2-val [--dynamic]
# VQAv2-testdev
GPUS=8 sh evaluate.sh <checkpoint> vqa-vqav2-testdev [--dynamic]
```

For the testdev set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission).

</details>

#### [OKVQA val](https://okvqa.allenai.org/)

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
GPUS=8 sh evaluate.sh <checkpoint> vqa-okvqa-val [--dynamic]
```

</details>

#### [TextVQA val](https://textvqa.org/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/textvqa && cd data/textvqa

# download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && unzip train_val_images.zip

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_questions.json
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val_llava.jsonl

cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
# without ocr tokens
GPUS=8 sh evaluate.sh <checkpoint> vqa-textvqa-val [--dynamic]
# with ocr tokens (hint: LLaVA use ocr tokens)
GPUS=8 sh evaluate.sh <checkpoint> vqa-textvqa-val-ocr [--dynamic]
```

</details>

#### [VizWiz val & test](https://vizwiz.org/tasks-and-datasets/vqa/)

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
GPUS=8 sh evaluate.sh <checkpoint> vqa-vizwiz-val [--dynamic]
# VizWiz test
GPUS=8 sh evaluate.sh <checkpoint> vqa-vizwiz-test [--dynamic]
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission).

</details>

#### [DocVQA val & test](https://www.docvqa.org/datasets)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/docvqa && cd data/docvqa

# download images and annotations
wget https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz --no-check-certificate # (optional)
wget https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz --no-check-certificate
wget https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz --no-check-certificate

# unzip files
tar -zxvf train.tar.gz
tar -zxvf val.tar.gz
tar -zxvf test.tar.gz

# download converted jsonl files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/docvqa/test.jsonl
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
# DocVQA-val
GPUS=8 sh evaluate.sh <checkpoint> vqa-docvqa-val [--dynamic]
# DocVQA-test
GPUS=8 sh evaluate.sh <checkpoint> vqa-docvqa-test [--dynamic]
```

For the test set, submit the results to the [evaluation server](https://rrc.cvc.uab.es/?ch=17).

</details>

#### [ChartQA test-human & test-augmented](https://aclanthology.org/2022.findings-acl.177/)

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
# test both ChartQA-test-human & ChartQA-test-augmented
GPUS=8 sh evaluate.sh <checkpoint> vqa-chartqa-test [--dynamic]
```

</details>

#### [GQA testdev](https://cs.stanford.edu/people/dorarad/gqa/about.html)

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
GPUS=8 sh evaluate.sh <checkpoint> vqa-gqa-testdev [--dynamic]
```

</details>

#### [OCRVQA val & test](https://ocr-vqa.github.io/)

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
GPUS=8 sh evaluate.sh <checkpoint> vqa-ocrvqa-val [--dynamic]
# OCRVQA-test
GPUS=8 sh evaluate.sh <checkpoint> vqa-ocrvqa-test [--dynamic]
```

</details>

#### [AI2D test](https://allenai.org/data/diagrams)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/ai2diagram && cd data/ai2diagram
# download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/ai2d_test_vlmevalkit.jsonl -O test_vlmevalkit.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/AI2D_TEST.zip && unzip AI2D_TEST.zip

# download images from Google drive (optional, provided by InternLM-XComposer)
# https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing
# images should be placed in `data/ai2diagram/ai2d/abc_images` and `data/ai2diagram/ai2d/images`
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> vqa-ai2d-test [--dynamic]
```

</details>

#### [ScienceQA test](https://github.com/lupantech/ScienceQA)

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
GPUS=8 sh evaluate.sh <checkpoint> scienceqa [--dynamic]
```

</details>

### Refer Expression Comprehension

#### RefCOCO/RefCOCO+/RefCOCO-g

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
GPUS=8 sh evaluate.sh <checkpoint> refcoco [--dynamic]
```

</details>

### MultiModal Benchmarks

#### [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/mme && cd data/mme

# 1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
# 2. Downloaded images to `MME_Benchmark_release_version`.

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

#### [MMBench dev & test](https://github.com/open-compass/mmbench/?tab=readme-ov-file)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/mmbench && cd data/mmbench

# download csv files of mmbench
wget http://opencompass.openxlab.space/utils/MMBench/CCBench_legacy.tsv
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
GPUS=8 sh evaluate.sh <checkpoint> mmbench-dev-en [--dynamic]
# mmbench_dev_cn_20231003
GPUS=8 sh evaluate.sh <checkpoint> mmbench-dev-cn [--dynamic]
# mmbench_test_en_20231003
GPUS=8 sh evaluate.sh <checkpoint> mmbench-test-en [--dynamic]
# mmbench_test_cn_20231003
GPUS=8 sh evaluate.sh <checkpoint> mmbench-test-cn [--dynamic]
# ccbench_dev
GPUS=8 sh evaluate.sh <checkpoint> ccbench-dev [--dynamic]
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission).

</details>

#### [POPE](https://github.com/AoiDragon/POPE/tree/main)

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
GPUS=8 sh evaluate.sh <checkpoint> pope [--dynamic]
```

</details>

#### [MMMU](MMMU_validation_240124181104.json)

<details>
<summary>Data Preparation</summary>

The evaluation code will automatically download the dataset from hugging face.

</details>

<details>
<summary>Evaluation</summary>

```bash
# dev set
GPUS=8 sh evaluate.sh <checkpoint> mmmu-dev [--dynamic]
# val set
GPUS=8 sh evaluate.sh <checkpoint> mmmu-val [--dynamic]
# test set
GPUS=8 sh evaluate.sh <checkpoint> mmmu-test [--dynamic]
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2179/overview).

</details>

#### [Tiny LVLM](https://github.com/OpenGVLab/Multi-Modality-Arena/tree/main/tiny_lvlm_evaluation)

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
GPUS=8 sh evaluate.sh <checkpoint> tiny_lvlm [--dynamic]
```

</details>

#### [LLaVA Bench](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild)

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

#### [MM-Vet](https://github.com/yuweihao/MM-Vet?tab=readme-ov-file)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/mm-vet && cd data/mm-vet
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/llava-mm-vet.jsonl
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> mmvet [--dynamic]
```

</details>

#### [MMVP](https://github.com/tsb0601/MMVP)

<details>
<summary>Data Preparation</summary>

```bash
cd data
git lfs install
git clone https://huggingface.co/datasets/MMVP/MMVP
git clone https://huggingface.co/datasets/MMVP/MMVP_VLM
cd ..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> mmvp [--dynamic]
```

</details>

#### [MathVista](https://github.com/lupantech/MathVista)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/MathVista && cd data/MathVista
wget https://huggingface.co/datasets/AI4Math/MathVista/raw/main/annot_testmini.json
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
export OPENAI_API_KEY='your-openai-key'
# testmini set
GPUS=8 sh evaluate.sh <checkpoint> mathvista-testmini [--dynamic]
# test set
GPUS=8 sh evaluate.sh <checkpoint> mathvista-test [--dynamic]
```

</details>

#### [SEED](https://github.com/AILab-CVC/SEED-Bench/)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/SEED && cd data/SEED
# 1. Follow the official instructions [Data Preparation for SEED-Bench-1](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md#data-preparation-for-seed-bench-1)
#    to download the images and the videos. Put images under `./data/SEED/SEED-Bench-image`.
# 2. Extract the video frame in the middle from the downloaded videos, and put them under `./data/SEED/SEED-Bench-image`.
#    LLaVA provided the script [`extract_video_frames.py`](../internvl_chat/tools/extract_video_frames.py) modified from the official one.

wget https://huggingface.co/OpenGVLab/InternVL/raw/main/seed.jsonl
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> seed [--dynamic]
```

</details>
