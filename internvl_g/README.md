# InternVL Stage-2 Pre-training

This folder contains the implementation of the InternVL for stage2 pre-training and retrieval fine-tuning.

## 🛠️ Install

- Clone this repository:

  ```bash
  git clone https://github.com/OpenGVLab/InternVL.git
  cd InternVL/internvl_g
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
  ```

## 📦 Data Preparation

**Pre-training**

Coming Soon

**Fine-tuning**

Three datasets need to be prepared: COCO Caption, Flickr30K, and NoCaps.

```shell
data
├── coco
│   ├── annotations
│   ├── test2017
│   ├── train2014
│   ├── train2017
│   ├── val2014
│   └── val2017
├── flickr30k
│   ├── flickr30k_cn_test.txt
│   ├── flickr30k_cn_train.txt
│   ├── flickr30k_test_karpathy.json
│   ├── flickr30k_test_karpathy.txt
│   ├── flickr30k_train_karpathy.txt
│   ├── flickr30k_val_karpathy.txt
│   └── Images
└── nocaps
    ├── images
    ├── nocaps_val_4500_captions_coco_format.json
    └── nocaps_val_4500_captions.json
```

## 🔥 Pre-training

Coming Soon

## 🔥 Retrieval Fine-tuning

To fine-tune InternVL on Flickr30K with 32 GPUs, run:

```bash
sh shell/finetune/internvl_stage2_finetune_flickr_364_bs1024_ep10.sh
```

To fine-tune InternVL on Flickr30K-CN with 32 GPUs, run:

```shell
sh shell/finetune/internvl_stage2_finetune_flickrcn_364_bs1024_ep10.sh
```

## 📊 Evaluation

### Zero-Shot Image Captioning

| model      | dataset                 | BLEU4 | METEOR | CIDEr |
| ---------- | ----------------------- | ----- | ------ | ----- |
| InternVL-G | COCO Karpathy test      | 37.1  | 30.1   | 128.2 |
| InternVL-G | Flickr30K Karpathy test | 27.0  | 25.3   | 79.2  |
| InternVL-G | NoCaps val              | 44.3  | 30.1   | 113.7 |

<details>
  <summary>[InternVL-G] COCO Karpathy test</summary>

```bash
sh evaluate.sh pretrained/internvl_14b_224px caption-coco
```

Expected results:

```
['coco', 'English caption:', 10.5974, dict_items([('Bleu_1', 0.7876323287981284), ('Bleu_2', 0.6353512494727918), ('Bleu_3', 0.49108984183589743), ('Bleu_4', 0.37062736733849205), ('METEOR', 0.30106315496945923), ('ROUGE_L', 0.5898249189475652), ('CIDEr', 1.281844384075423)])]
```

</details>

<details>
  <summary>[InternVL-G] Flickr30K Karpathy test</summary>

```bash
['flickr30k', 'English caption:', 10.666, dict_items([('Bleu_1', 0.7182900534357628), ('Bleu_2', 0.5353390037921949), ('Bleu_3', 0.3834462132295285), ('Bleu_4', 0.2702131471765472), ('METEOR', 0.25263515267930103), ('ROUGE_L', 0.5305876871149064), ('CIDEr', 0.7919734768328237)])]
```

Expected results:

```
sh evaluate.sh pretrained/internvl_14b_224px caption-flickr30k
```

</details>

<details>
  <summary>[InternVL-G] NoCaps val</summary>

```bash
sh evaluate.sh pretrained/internvl_14b_224px caption-nocaps
```

Expected results:

```
['nocaps', 'English caption:', 10.463111111111111, dict_items([('Bleu_1', 0.8518290482155187), ('Bleu_2', 0.7165227921485106), ('Bleu_3', 0.5733723839888316), ('Bleu_4', 0.44268902150723105), ('METEOR', 0.30078174807736896), ('ROUGE_L', 0.6070208063052156), ('CIDEr', 1.1371742045267772)])]
```

</details>

### Fine-tuned Image-Text Retrieval

<table>
  <tr  align=center>
      <td rowspan="3" align=center><b>model</b></td>
      <td colspan="6" align=center><b>Flickr30K</b></td>
      <td colspan="6" align=center><b>Flickr30K-CN</b></td>
      <td rowspan="3" align=center><b>avg</b></td>

</tr>
   <tr  align=center>
      <td colspan="3" align=center><b>image-to-text</b></td>
      <td colspan="3" align=center><b>text-to-image</b></td>
       <td colspan="3" align=center><b>image-to-text</b></td>
      <td colspan="3" align=center><b>text-to-image</b></td>
   </tr>
   <tr>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
   </tr>

<tr align=center>
      <td>InternVL-C-FT</td>
      <td>97.2</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>88.5</td>
      <td>98.4</td>
      <td>99.2</td>
      <td>96.5</td>
      <td>99.9</td>
      <td>100.0</td>
      <td>85.2</td>
      <td>97.0</td>
      <td>98.5</td>
      <td>96.7</td>
   </tr>
<tr align=center>
      <td>InternVL-G-FT</td>
      <td>97.9</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>89.6</td>
      <td>98.6</td>
      <td>99.2</td>
      <td>96.9</td>
      <td>99.9</td>
      <td>100.0</td>
      <td>85.9</td>
      <td>97.1</td>
      <td>98.7</td>
      <td>97.0</td>
   </tr>

</table>

<details>
  <summary>[InternVL-C-FT] Flickr30K</summary>

```bash
cd ../clip_benchmark/
CUDA_VISIBLE_DEVICES=0 python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_c_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickr_364_bs1024_ep10/ --output result_ft.json
```
Expected results:

```
{"dataset": "flickr30k", "model": "internvl_c_retrieval_hf", "pretrained": "./work_dirs/internvl_stage2_finetune_flickr_364_bs1024_ep10", "task": "zeroshot_retrieval",
"metrics": {"image_retrieval_recall@1": 0.8853999972343445, "text_retrieval_recall@1": 0.972000002861023,
"image_retrieval_recall@5": 0.9836000204086304, "text_retrieval_recall@5": 1.0,
"image_retrieval_recall@10": 0.9923999905586243, "text_retrieval_recall@10": 1.0}, "language": "en"}
```
</details>

<details>
  <summary>[InternVL-C-FT] Flickr30K-CN</summary>

```bash
cd ../clip_benchmark/
CUDA_VISIBLE_DEVICES=0 python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_c_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/ --output result_ft.json
```

Expected results:

```
{"dataset": "flickr30k", "model": "internvl_c_retrieval_hf", "pretrained": "./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10", "task": "zeroshot_retrieval",
"metrics": {"image_retrieval_recall@1": 0.8521999716758728, "text_retrieval_recall@1": 0.9649999737739563,
"image_retrieval_recall@5": 0.9697999954223633, "text_retrieval_recall@5": 0.9990000128746033,
"image_retrieval_recall@10": 0.9854000210762024, "text_retrieval_recall@10": 1.0}, "language": "cn"}
```
</details>

<details>
  <summary>[InternVL-G-FT] Flickr30K</summary>

```bash
cd ../clip_benchmark/
CUDA_VISIBLE_DEVICES=0 python3 clip_benchmark/cli.py eval --model_type internvl --language "en" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickr_364_bs1024_ep10/ --output result_ft.json
```

Expected results:

```
{"dataset": "flickr30k", "model": "internvl_g_retrieval_hf", "pretrained": "./work_dirs/internvl_stage2_finetune_flickr_364_bs1024_ep10", "task": "zeroshot_retrieval",
"metrics": {"image_retrieval_recall@1": 0.895799994468689, "text_retrieval_recall@1": 0.9789999723434448,
"image_retrieval_recall@5": 0.9861999750137329, "text_retrieval_recall@5": 1.0,
"image_retrieval_recall@10": 0.9922000169754028, "text_retrieval_recall@10": 1.0}, "language": "en"}
```
</details>

<details>
  <summary>[InternVL-G-FT] Flickr30K-CN</summary>

```bash
cd ../clip_benchmark/
CUDA_VISIBLE_DEVICES=0 python3 clip_benchmark/cli.py eval --model_type internvl --language "cn" --task "zeroshot_retrieval" \
     --dataset "flickr30k" --dataset_root ./data/flickr30k --model internvl_g_retrieval_hf \
     --pretrained ./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10/ --output result_ft.json
```

Expected results:

```
{"dataset": "flickr30k", "model": "internvl_g_retrieval_hf", "pretrained": "./work_dirs/internvl_stage2_finetune_flickrcn_364_bs1024_ep10", "task": "zeroshot_retrieval",
"metrics": {"image_retrieval_recall@1": 0.8587999939918518, "text_retrieval_recall@1": 0.968999981880188,
"image_retrieval_recall@5": 0.9714000225067139, "text_retrieval_recall@5": 0.9990000128746033,
"image_retrieval_recall@10": 0.9865999817848206, "text_retrieval_recall@10": 1.0}, "language": "cn"}
```
</details>
