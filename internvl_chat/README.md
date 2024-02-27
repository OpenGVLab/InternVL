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
| InternViT-6B-448px-V1-2 | ViT  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2) | 11.1 GB |
| Nous-Hermes-2-Yi-34B    | LLM  | 🤗 [HF link](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) | 65.0 GB |

Please download the above model weights and place them in the `pretrained/` folder.

```sh
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternViT-6B-448px-V1-2 --local-dir intern_vit_6b_448px_v1_2
huggingface-cli download --resume-download --local-dir-use-symlinks False NousResearch/Nous-Hermes-2-Yi-34B --local-dir Nous-Hermes-2-Yi-34B
```

The directory structure is:

```sh
pretrained
│── intern_vit_6b_448px_v1_2/
└── Nous-Hermes-2-Yi-34B/
```

## 🔥 Supervised Fine-tuning

### Prepare Training Datasets

Inspired by LLaVA-NeXT, we adopted a data-efficient SFT strategy to train InternVL-Chat-V1.2, utilizing approximately 1.2M of visual instruction tuning samples in total, all of which are fully open-source. In a macro sense, we build upon [ShareGPT-4V](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md#prepare-images) and additionally integrate [LLaVA-ZH](https://huggingface.co/datasets/openbmb/llava_zh), [DVQA](https://github.com/kushalkafle/DVQA_dataset), [ChartQA](https://github.com/vis-nlp/ChartQA), [AI2D](https://allenai.org/data/diagrams), [DocVQA](https://www.docvqa.org/datasets), [GeoQA+](https://github.com/SCNU203/GeoQA-Plus), and [SynthDoG-EN](https://huggingface.co/datasets/naver-clova-ix/synthdog-en). Most of the data remains consistent with LLaVA-NeXT.

First, download the [annotation files](https://huggingface.co/OpenGVLab/InternVL/resolve/main/playground.zip) and place them in the `playground/` folder.

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
├── sharegpt4v_instruct_gpt4-vision_cap100k.jsonl
├── llava_instruct_150k_zh.jsonl
├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl
├── dvqa_train_200k.jsonl
├── chartqa_train_18k.jsonl
├── ai2d_train_12k.jsonl
├── docvqa_train_10k.jsonl
├── geoqa+.jsonl
├── synthdog_en.jsonl
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
PARTITION='your partition' GPUS=32 PER_DEVICE_BATCH_SIZE=8 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_finetune.sh
# using 64 GPUs
PARTITION='your partition' GPUS=64 PER_DEVICE_BATCH_SIZE=8 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_finetune.sh
```

The hyperparameters used for fine-tuning are listed in the following table. And, you can view the training logs in tensorboard at [here](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2/tensorboard).

| Hyperparameter     | Trainable Param  | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| ------------------ | ---------------- | ----------------- | ------------- | ------ | ---------- | ------------ |
| InternVL-Chat-V1.2 | 40B (full model) | 512               | 1e-5          | 1      | 2048       | 0.05         |

## Continue Fine-tune

You can continue to fine-tune the checkpoint from the previous training process use this [script](./shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_finetune_continue.sh).

Before fine-tuning, you should set the `--meta_path` in to your custom meta file of training data.

```sh
# using 16 GPUs, fine-tune the full LLM
PARTITION='your partition' GPUS=16 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_finetune_continue.sh
# using 2 GPUs, fine-tune the LoRA
CUDA_VISIBLE_DEVICES=0,1 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_finetune_continue_lora.sh
```

## 📊 Evaluation

**MultiModal Benchmark**

\* Training set observed.

| name                                                                                        | model size | MathVista<br>(testmini) | MMB<br>(dev/test) | MMB−CN<br>(dev/test) | MMMU<br>(val/test)                                                                 | CMMMU<br>(val/test) | MMVP | MME            | POPE | Tiny LVLM | SEEDv1<br>(image) | LLaVA Wild | MM−Vet |
| ------------------------------------------------------------------------------------------- | ---------- | ----------------------- | ----------------- | -------------------- | ---------------------------------------------------------------------------------- | ------------------- | ---- | -------------- | ---- | --------- | ----------------- | ---------- | ------ |
| [InternVL−Chat−V1.1](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-1)           | 19B        | 34.5                    | 76.7 / 75.4       | 71.9 / 70.3          | 39.1 / 35.3                                                                        | 34.8 / 34.0         | 44.7 | 1675.1 / 348.6 | 87.1 | 343.2     | 73.2              | 73.2       | 46.7   |
| [InternVL−Chat−V1.2](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2)           | 40B        | 47.7                    | 81.4 / 82.2       | 79.5 / 81.2          | 51.6 / [46.2](https://eval.ai/web/challenges/challenge-page/2179/leaderboard/5377) | TODO                | 56.7 | 1672.1 / 509.3 | 88.0 | 350.3     | 75.6              | 85.0       | 48.9   |
| [InternVL−Chat−V1.2−Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2-Plus) | 40B        | 59.9                    | 83.4 / 83.8       | 81.6 / 82.0          | 50.3 / 45.6                                                                        | TODO                | 58.7 | 1623.6 / 550.7 | 88.7 | 353.9     | 76.4              | 84.6       | 47.9   |

**Image Captioning & Visual Question Answering**

\* Training set observed.

| name                                                                                        | model size | COCO<br>(test) | Flickr30K<br>(test) | NoCaps<br>(val) | VQAv2<br>(testdev) | OKVQA<br>(val) | TextVQA<br>(val) | VizWiz<br>(val/test) | AI2D<br>(test) | GQA<br>(test) | ScienceQA<br>(image) |
| ------------------------------------------------------------------------------------------- | ---------- | -------------- | ------------------- | --------------- | ------------------ | -------------- | ---------------- | -------------------- | -------------- | ------------- | -------------------- |
| [InternVL−Chat−V1.1](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-1)           | 19B        | 142.2\*        | 85.3                | 120.8           | 80.9\*             | 64.1\*         | 65.9             | 59.0 / 57.3          | 72.2\*         | 62.5\*        | 90.1\*               |
| [InternVL−Chat−V1.2](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2)           | 40B        | 113.9          | 92.4                | 112.5           | -                  | 62.5\*         | 69.7             | 61.9 / 60.0          | 77.1\*         | 64.0\*        | 83.3                 |
| [InternVL−Chat−V1.2−Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2-Plus) | 40B        | 143.4\*        | 90.5                | 125.8           | -                  | 67.6\*         | 71.3\*           | 61.3 / -             | 78.2\*         | 66.9\*        | 98.1\*               |

- We found that incorrect images were used for training and testing in `AI2D`, meaning that for problems where `abcLabel` is True, `abc_images` were not utilized. We have now corrected the images used for testing, but the results may still be somewhat lower as a consequence.

**Visual Grounding**

| name                                                                                        | model size | RefCOCO<br>(val) | RefCOCO<br>(testA) | RefCOCO<br>(testB) | RefCOCO+<br>(val) | RefCOCO+<br>(testA) | RefCOCO+<br>(testB) | RefCOCO−g<br>(val) | RefCOCO−g<br>(test) |
| ------------------------------------------------------------------------------------------- | ---------- | ---------------- | ------------------ | ------------------ | ----------------- | ------------------- | ------------------- | ------------------ | ------------------- |
| [InternVL−Chat−V1.1](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-1)           | 19B        | 84.7             | 89.9               | 78.6               | 78.5              | 85.6                | 70.1                | 81.0               | 81.4                |
| [InternVL−Chat−V1.2](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2)           | 40B        | 74.4             | 80.3               | 66.5               | 70.7              | 77.6                | 62.0                | 69.2               | 70.0                |
| [InternVL−Chat−V1.2−Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2-Plus) | 40B        | 90.2             | 93.4               | 85.5               | 85.3              | 90.4                | 79.7                | 88.5               | 88.8                |

## 📊 Evaluation (Legacy Models)

| model         | QLLaMA | LLM          | res | COCO  | Flickr | NoCaps | VQAv2 | GQA  | VizWiz | TextVQA | MME    | POPE | Download |
| ------------- | ------ | ------------ | --- | ----- | ------ | ------ | ----- | ---- | ------ | ------- | ------ | ---- | -------- |
| InternVL-Chat | ✔️     | frozen V-7B  | 224 | 141.4 | 89.7   | 120.5  | 72.3  | 57.7 | 44.5   | 42.1    | 1298.5 | 85.2 | TODO     |
| InternVL-Chat | ✔️     | frozen V-13B | 224 | 142.4 | 89.9   | 123.1  | 71.7  | 59.5 | 54.0   | 49.1    | 1317.2 | 85.4 | TODO     |
| InternVL-Chat | ✔️     | V-13B        | 336 | 146.2 | 92.2   | 126.2  | 81.2  | 66.6 | 58.5   | 61.5    | 1586.4 | 87.6 | TODO     |

## ❓ How to Evaluate

Please prepare the data according to the following directory structure.

<details>
<summary>Directory Structure</summary>

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
│   │   ├── abc_images/
│   │   └── images/
│   └── test.jsonl
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
├── mmmu
│   ├── Accounting/
│   ├── ...
│   └── Sociology
├── mm-vet
│   └── images/
├── MMVP
│   ├── MMVP Images/
│   ├── Questions.csv
│   └── Questions.xlsx
├── MMVP_VLM
│   ├── MLLM_VLM Images/
│   └── Questions.csv
├── MathVista
│   ├── annot_testmini.json
│   └── AI4Math___math_vista/
├── SEED
│   ├── SEED-Bench.json
│   ├── SEED-Bench-image/
│   └── SEED-Bench-video-image-1/
```

</details>

### Image Caption

#### [COCO karpathy test](https://cocodataset.org/)

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
GPUS=8 sh evaluate.sh <checkpoint> caption-coco
```

</details>

#### [Flickr30K karpathy test](https://bryanplummer.com/Flickr30kEntities/)

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
GPUS=8 sh evaluate.sh <checkpoint> caption-flickr30k
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
GPUS=8 sh evaluate.sh <checkpoint> caption-nocaps
```

</details>

### General VQA

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
GPUS=8 sh evaluate.sh <checkpoint> vqa-vqav2-val
# VQAv2-testdev
GPUS=8 sh evaluate.sh <checkpoint> vqa-vqav2-testdev
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
GPUS=8 sh evaluate.sh <checkpoint> vqa-okvqa-val
```

</details>

#### [TextVQA val](https://textvqa.org/)

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
GPUS=8 sh evaluate.sh <checkpoint> vqa-textvqa-val
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
GPUS=8 sh evaluate.sh <checkpoint> vqa-vizwiz-val
# VizWiz test
GPUS=8 sh evaluate.sh <checkpoint> vqa-vizwiz-test
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission).

</details>

#### [DocVQA val & test](https://www.docvqa.org/datasets)

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
GPUS=8 sh evaluate.sh <checkpoint> vqa-docvqa-val
# DocVQA-test
GPUS=8 sh evaluate.sh <checkpoint> vqa-docvqa-test
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
# ChartQA-test-human
GPUS=8 sh evaluate.sh <checkpoint> vqa-chartqa-test-human
# ChartQA-test-augmented
GPUS=8 sh evaluate.sh <checkpoint> vqa-chartqa-test-augmented
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
GPUS=8 sh evaluate.sh <checkpoint> vqa-gqa-testdev
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
GPUS=8 sh evaluate.sh <checkpoint> vqa-ocrvqa-val
# OCRVQA-test
GPUS=8 sh evaluate.sh <checkpoint> vqa-ocrvqa-test
```

</details>

#### [AI2Diagram test](https://allenai.org/data/diagrams)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/ai2diagram && cd data/ai2diagram
# download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/ai2d_test.jsonl -O test.jsonl

# download images from Google drive (provided by InternLM-XComposer)
# https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing
# images should be placed in `data/ai2diagram/ai2d/abc_images` and `data/ai2diagram/ai2d/images`
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> vqa-ai2d-test
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
GPUS=8 sh evaluate.sh <checkpoint> scienceqa
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
GPUS=8 sh evaluate.sh <checkpoint> refcoco
```

</details>

### MultiModal Dialogue

#### [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

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

#### [MMBench dev & test](https://github.com/open-compass/mmbench/?tab=readme-ov-file)

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
GPUS=8 sh evaluate.sh <checkpoint> mmbench-dev-en
# mmbench_dev_cn_20231003
GPUS=8 sh evaluate.sh <checkpoint> mmbench-dev-cn
# mmbench_test_en_20231003
GPUS=8 sh evaluate.sh <checkpoint> mmbench-test-en
# mmbench_test_cn_20231003
GPUS=8 sh evaluate.sh <checkpoint> mmbench-test-cn
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
GPUS=8 sh evaluate.sh <checkpoint> pope
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
GPUS=8 sh evaluate.sh <checkpoint> mmmu-dev
# val set
GPUS=8 sh evaluate.sh <checkpoint> mmmu-val
# test set
GPUS=8 sh evaluate.sh <checkpoint> mmmu-test
```

For the test set, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2179/overview).

</details>

#### CMMMU

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
GPUS=8 sh evaluate.sh <checkpoint> tiny_lvlm
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
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> mmvet
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
GPUS=8 sh evaluate.sh <checkpoint> mmvp
```

</details>

#### [MathVista](https://github.com/lupantech/MathVista)

<details>
<summary>Data Preparation</summary>

```bash
mkdir -p data/MathVista && cd data/MathVista
# Execute the following python code
# from datasets import load_dataset
# dataset = load_dataset("AI4Math/MathVista")
# dataset.save_to_disk('./MathVista')
wget https://huggingface.co/datasets/AI4Math/MathVista/raw/main/annot_testmini.json
cd ../..
```

</details>

<details>
<summary>Evaluation</summary>

```bash
# testmini set
GPUS=8 sh evaluate.sh <checkpoint> mathvista-testmini
# test set
GPUS=8 sh evaluate.sh <checkpoint> mathvista-test
```

</details>

#### [SEED](https://github.com/AILab-CVC/SEED-Bench/)

<details>
<summary>Data Preparation</summary>

1. Follow the official instructions [Data Preparation for SEED-Bench-1](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md#data-preparation-for-seed-bench-1) to download the images and the videos. Put images under `./playground/data/eval/seed_bench/SEED-Bench-image`.
2. Extract the video frame in the middle from the downloaded videos, and put them under `./playground/data/eval/seed_bench/SEED-Bench-video-image`. We provide our script `extract_video_frames.py` modified from the official one.

</details>

<details>
<summary>Evaluation</summary>

```bash
GPUS=8 sh evaluate.sh <checkpoint> seed
```

</details>
