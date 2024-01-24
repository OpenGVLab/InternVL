# InternVL-Chat

This folder contains the implementation of the InternVL-Chat.

## 🛠️ Installation

> If you have already installed the environment as per the instructions in other folders, you can skip this section.

- Clone this repository:

  ```bash
  git clone https://github.com/OpenGVLab/InternVL.git
  cd InternVL/internvl_chat
  ```

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n internvl python=3.9 -y
  conda activate internvl
  ```

- Install `PyTorch>=2.0` and `torchvision>=0.15.2` with `CUDA>=11.6`:

  For examples, to install `torch==2.0.1` with `CUDA==11.8`:

  ```bash
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  # or
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install `flash-attn==0.2.8` :

  If you want to fully replicate my results, please install `v0.2.8`, otherwise install the latest version.

  This is because different versions of flash attention yield slight differences in results.

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v0.2.8
  python setup.py install
  ```

- Install `timm==0.6.11` and `mmcv-full==1.6.2`:

  ```bash
  pip install -U openmim
  pip install timm==0.6.11
  mim install mmcv-full==1.6.2
  ```

- Install `apex`:

  ```bash
  git clone https://github.com/NVIDIA/apex.git
  git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # https://github.com/NVIDIA/apex/issues/1735
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  ```

- Install other requirements:

  ```bash
  pip install opencv-python termcolor yacs pyyaml scipy
  pip install deepspeed==0.10.0
  pip install pycocoevalcap tqdm
  ```

## 📦 Model Preparation

| model name         | type        | download                                                          |  size   |
| ------------------ | ----------- | ----------------------------------------------------------------- | :-----: |
| InternVL-14B-224px | huggingface | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-14B-224px) | 27.7 GB |
| Vicuna-7B-v1.5     | huggingface | 🤗 [HF link](https://huggingface.co/lmsys/vicuna-7b-v1.5)         | 13.5 GB |
| Vicuna-13B-v1.5    | huggingface | 🤗 [HF link](https://huggingface.co/lmsys/vicuna-13b-v1.5)        | 26.1 GB |

Please download the above model weights and place them in the `pretrained/` folder.

```sh
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-14B-224px --local-dir internvl_14b_224px
huggingface-cli download --resume-download --local-dir-use-symlinks False lmsys/vicuna-13b-v1.5 --local-dir vicuna-13b-v1.5
huggingface-cli download --resume-download --local-dir-use-symlinks False lmsys/vicuna-7b-v1.5 --local-dir vicuna-7b-v1.5
```

The directory structure is:

```sh
pretrained
│── internvl_14b_224px/
│── vicuna-13b-v1.5/
└── vicuna-7b-v1.5/
```

## 🔥 Supervised Fine-tuning

Coming Soon

## 📊 Evaluation (English Models)

| model         | QLLaMA | LLM          | res | COCO  | Flickr | NoCaps | VQAv2 | GQA  | VizWiz | TextVQA | MME    | POPE | Download |
| ------------- | ------ | ------------ | --- | ----- | ------ | ------ | ----- | ---- | ------ | ------- | ------ | ---- | -------- |
| InternVL-Chat | ✔️     | frozen V-7B  | 224 | 141.4 | 89.7   | 120.5  | 72.3  | 57.7 | 44.5   | 42.1    | 1298.5 | 85.2 | TODO     |
| InternVL-Chat | ✔️     | frozen V-13B | 224 | 142.4 | 89.9   | 123.1  | 71.7  | 59.5 | 54.0   | 49.1    | 1317.2 | 85.4 | TODO     |
| InternVL-Chat | ✔️     | V-13B        | 336 | 146.2 | 92.2   | 126.2  | 81.2  | 66.6 | 58.5   | 61.5    | 1586.4 | 87.6 | TODO     |

## 📊 Evaluation (Chinese Models)

**MultiModal Benchmark**

| model                                                                             | MME            | MMB<sub>dev/test</sub> | MMB-CN<sub>dev/test</sub> | POPE | MMMU | CMMMU | Tiny LVLM | LLaVA<sub>bench</sub> |
| --------------------------------------------------------------------------------- | -------------- | ---------------------- | ------------------------- | ---- | ---- | ----- | --------- | --------------------- |
| [InternVL-Chat-V1.1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1.1-Chinese) | 1672.4 / 341.1 | 76.6 / 75.4            | 71.5 / 70.1               | 87.2 |      |       | 344.5     | 76.3                  |

**Visual Question Answering**

| model                                                                             | VQAv2<sub>test</sub> | OKVQA<sub>val</sub> | TextVQA<sub>val</sub> | VizWiz<sub>val/test</sub> | AI2D<sub>test</sub> | GQA<sub>test</sub> | SQA<sub>test</sub> |
| --------------------------------------------------------------------------------- | -------------------- | ------------------- | --------------------- | ------------------------- | ------------------- | ------------------ | ------------------ |
| [InternVL-Chat-V1.1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1.1-Chinese) | 80.9                 | 64.2                | 65.8                  | 58.3 / 57.3               | 70.23               | 62.4               | 91.2               |

**Image Captioning**

| model                                                                             | COCO<sub>test</sub> | Flickr30K<sub>test</sub> | NoCaps<sub>val</sub> |
| --------------------------------------------------------------------------------- | ------------------- | ------------------------ | -------------------- |
| [InternVL-Chat-V1.1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1.1-Chinese) | 141.8               | 84.3                     | 120.4                |

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
├── mmmu
│   ├── Accounting/
│   ├── ...
│   └── Sociology
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> caption-coco
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> caption-flickr30k
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> caption-nocaps
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vqav2-val
# VQAv2-testdev
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vqav2-testdev
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-okvqa-val
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-textvqa-val
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vizwiz-val
# VizWiz test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-vizwiz-test
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-docvqa-val
# DocVQA-test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-docvqa-test
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-chartqa-test-human
# ChartQA-test-augmented
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-chartqa-test-augmented
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-gqa-testdev
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-ocrvqa-val
# OCRVQA-test
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> vqa-ocrvqa-test
```

</details>

#### [AI2Diagram test](https://allenai.org/data/diagrams)

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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> scienceqa
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> refcoco
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> pope
```

</details>

#### MMMU

<details>
<summary>Data Preparation</summary>

The evaluation code will automatically download the dataset from hugging face.

</details>

<details>
<summary>Evaluation</summary>

```bash
# dev set
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> mmmu-dev
# val set
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> mmmu-val
# test set
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> mmmu-test
```

Then, submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/2179/overview).

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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh evaluate.sh <checkpoint> tiny_lvlm
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
