# How to Evaluate InternVL-Chat-V1-5?

In this tutorial, we will provide a detailed guide on how to replicate the results presented in the InternVL 1.5 technical report.

The results are shown in the table below.

_If you encounter any difficulties while testing according to this guide, please let me know. Thank you._

> Note that if you are aiming for an exact replication, please use this code repository and follow the testing methods outlined below; otherwise, using the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) would be more convenient.

![image](https://github.com/OpenGVLab/InternVL/assets/23737120/8b62d429-c689-426a-9267-2727b6430b6e)

## Model Preparation

| model name         | type | download                                                          | #param |
| ------------------ | ---- | ----------------------------------------------------------------- | :----: |
| InternVL-Chat-V1-5 | MLLM | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) | 25.5B  |

Please download the above model weights and place them in the `pretrained/` folder.

```sh
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-5 --local-dir InternVL-Chat-V1-5
```

The directory structure is:

```
pretrained
└── InternVL-Chat-V1-5
```

## OCR-related Benchmarks

Our tests will be divided into three parts. First, we will focus on OCR-related datasets, including DocVQA, ChartQA, InfoVQA, TextVQA, and OCRBench. Next, let's proceed to test each dataset one by one.

### DocVQA val & test

<details>
<summary>click to expand</summary>

1. Download the DocVQA dataset using the following instructions:

   ```shell
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

2. After preparation is complete, the directory structure is:

   ```
   data
    ├── docvqa
    │   ├── test
    │   ├── test.jsonl
    │   ├── train
    │   ├── train.jsonl
    │   ├── val
    │   └── val.jsonl
   ```

3. Test the model with the following commands:

   We use a maximum of `18 tiles` to test the DocVQA dataset.

   ```shell
   # evaluation on the val set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-docvqa-val --dynamic --max-num 18
   # evaluation on the test set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-docvqa-test --dynamic --max-num 18
   ```

   The result of the validation set is:

   ```
   Overall ANLS: 0.9049
   ```

   For the test set, the test results need to be submitted to the [testing server](https://rrc.cvc.uab.es/?ch=17&com=tasks).

</details>

### ChartQA test

<details>
<summary>click to expand</summary>

1. Download the ChartQA dataset using the following instructions:

   ```shell
   mkdir -p data/chartqa && cd data/chartqa

   # download images from https://drive.google.com/file/d/1Lm_w6zeET1Hyl_9ks6w5nEsgpoyPHalV/view

   # download converted files
   wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_human.jsonl
   wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/train_augmented.jsonl
   wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_human.jsonl
   wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/chartqa/test_augmented.jsonl

   cd ../..
   ```

2. After preparation is complete, the directory structure is:

   ```
   data
    ├── chartqa
    │   ├── ChartQA Dataset
    │   │    ├── test
    │   │    ├── train
    │   │    └── val
    │   ├── test_augmented.jsonl
    │   ├── test_human.jsonl
    │   ├── train_augmented.jsonl
    │   └── train_human.jsonl
   ```

3. Test the model with the following commands:

   We use a maximum of `12 tiles` to test the ChartQA dataset.

   ```shell
   # evaluation on the test set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-chartqa-test --dynamic --max-num 12
   ```

   The result of the test set is:

   ```
   ['chartqa_test_human', {'relaxed_accuracy': 0.736}]
   ['chartqa_test_augmented', {'relaxed_accuracy': 0.9408}]
   # the average score = (73.6 + 94.08) / 2 = 83.8
   ```

</details>

### InfoVQA val & test

<details>
<summary>click to expand</summary>

1. Download the InfoVQA dataset using the following instructions:

   ```shell
   mkdir -p data/infographicsvqa && cd data/infographicsvqa

   # download images and annotations from https://rrc.cvc.uab.es/?ch=17&com=downloads
   # infographicsVQA_test_v1.0.json, infographicsVQA_val_v1.0_withQT.json, infographicVQA_train_v1.0.json

   # download converted files
   wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_val.jsonl -O val.jsonl
   wget https://huggingface.co/OpenGVLab/InternVL/raw/main/infographicsvqa_test.jsonl -O test.jsonl

   cd ../..
   ```

2. After preparation is complete, the directory structure is:

   ```
   data
    ├── infographicsvqa
    │   ├── infographicsvqa_images
    │   ├── infographicsVQA_test_v1.0.json
    │   ├── infographicsVQA_val_v1.0_withQT.json
    │   ├── infographicVQA_train_v1.0.json
    │   ├── test.jsonl
    │   └── val.jsonl
   ```

3. Test the model with the following commands:

   We use a maximum of `24 tiles` to test the InfoVQA dataset.

   ```shell
   # evaluation on the val set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-infovqa-val --dynamic --max-num 24
   # evaluation on the test set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-infovqa-test --dynamic --max-num 24
   ```

   The result of the val set is:

   ```
   Overall ANLS: 0.7235
   ```

   For the test set, the test results need to be submitted to the [testing server](https://rrc.cvc.uab.es/?ch=17&com=tasks).

</details>

### TextVQA val

<details>
<summary>click to expand</summary>

1. Download the TextVQA dataset using the following instructions:

   ```shell
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

2. After preparation is complete, the directory structure is:

   ```
   data
    ├── textvqa
    │   ├── textvqa_train_annotations.json
    │   ├── textvqa_train.jsonl
    │   ├── textvqa_train_questions.json
    │   ├── textvqa_val_annotations.json
    │   ├── textvqa_val.jsonl
    │   ├── textvqa_val_llava.jsonl
    │   ├── textvqa_val_questions.json
    │   └── train_images
   ```

3. Test the model with the following commands:

   We use a maximum of `24 tiles` to test the TextVQA dataset.

   ```shell
   # evaluation on the val set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-textvqa-val --dynamic --max-num 24
   ```

   The result of the val set is:

   ```
   ['pretrained/InternVL-Chat-V1-5', 'textvqa_val', 0.8061000000000043]
   ```

</details>

### OCRBench

<details>
<summary>click to expand</summary>

Please use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for the test of OCRBench.

The command to test InternVL-Chat-V1-5 on OCRBench using VLMEvalKit is:

```
torchrun --nproc-per-node=8 run.py --data OCRBench --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 00:28:29,681 - Evaluation - INFO - Score:
2024-04-29 00:28:29,681 - Evaluation - INFO - Text Recognition:238
2024-04-29 00:28:29,681 - Evaluation - INFO - Scene Text-centric VQA:178
2024-04-29 00:28:29,681 - Evaluation - INFO - Doc-oriented VQA:151
2024-04-29 00:28:29,681 - Evaluation - INFO - Key Information Extraction:153
2024-04-29 00:28:29,681 - Evaluation - INFO - Handwritten Mathematical Expression Recognition:4
2024-04-29 00:28:29,681 - Evaluation - INFO - Final Score:724
2024-04-29 00:28:29,681 - Evaluation - INFO - Final Score Norm:72.4
```

</details>

## General Multimodal Benchmarks

Next, we will test InternVL-Chat-V1.5 using 10 general multimodal benchmarks, which include MME, RealWorldQA, AI2D, MMMU, MMBench-EN, MMBench-CN, CCBench, MMVet, SEED, and HallusionBench.

### MME

<details>
<summary>click to expand</summary>

1. Download the MME dataset using the following instructions:

   ```shell
   mkdir -p data/mme && cd data/mme

   # 1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
   # 2. Downloaded images to `MME_Benchmark_release_version`.

   cd ../..
   ```

2. After preparation is complete, the directory structure is:

   ```
   data
    ├── mme
    │   └── MME_Benchmark_release_version
   ```

3. Single-GPU inference and evaluate:

   We use a maximum of `12 tiles` to test the MME dataset.

   ```shell
   # evaluation on the val set
   GPUS=1 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mme --dynamic --max-num 12
   ```

   The result of MME is:

   ```
   total score: 1658.3683473389356

      existence  score: 190.0
      count  score: 175.0
      position  score: 171.66666666666669
      color  score: 178.33333333333331
      posters  score: 173.8095238095238
      celebrity  score: 142.05882352941177
      scene  score: 156.5
      landmark  score: 179.5
      artwork  score: 144.0
      OCR  score: 147.5


   =========== Cognition ===========
   total score: 533.5714285714286

      commonsense_reasoning  score: 133.57142857142858
      numerical_calculation  score: 117.5
      text_translation  score: 185.0
      code_reasoning  score: 97.5

   # 1658.3683473389356 + 533.5714285714286 = 2191.939775910364
   ```

</details>

### RealWorldQA

<details>
<summary>click to expand</summary>

Please use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for the test of RealWorldQA.

The command to test InternVL-Chat-V1-5 on RealWorldQA using VLMEvalKit is:

```
torchrun --nproc-per-node=8 run.py --data RealWorldQA --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 00:35:13,282 - Evaluation - INFO - Score:
2024-04-29 00:35:13,282 - Evaluation - INFO -   split   Overall
0  none  0.660131
```

</details>

### AI2D test

<details>
<summary>click to expand</summary>

1. Download the AI2D dataset using the following instructions:

   ```shell
   mkdir -p data/ai2diagram && cd data/ai2diagram
   # download converted files
   wget https://huggingface.co/OpenGVLab/InternVL/raw/main/ai2d_test_vlmevalkit.jsonl -O test_vlmevalkit.jsonl
   wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/AI2D_TEST.zip && unzip AI2D_TEST.zip

   # download images from Google drive (optional, provided by InternLM-XComposer)
   # https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing
   # images should be placed in `data/ai2diagram/ai2d/abc_images` and `data/ai2diagram/ai2d/images`
   cd ../..
   ```

2. After preparation is complete, the directory structure is:

   ```
   data
    ├── ai2diagram
    │   ├── test_vlmevalkit.jsonl
    │   ├── ai2d # (optional)
    │   │    ├── abc_images
    │   │    └── images
    │   └── AI2D_TEST
   ```

3. Test the model with the following commands:

   We use a maximum of `6 tiles` to test the AI2D dataset.

   ```shell
   # evaluation on the test set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 vqa-ai2d-test --dynamic
   ```

   The result of AI2D is:

   ```
   ai2diagram_test {'accuracy': 0.8073186528497409}
   ```

</details>

### MMMU val

<details>
<summary>click to expand</summary>

1. The evaluation code will automatically download the dataset from HuggingFace.

2. Test the model with the following commands:

   ```
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmmu-val --dynamic
   ```

   The result of MMMU val is:

   ```
   {'Overall-Art and Design': {'num': 120, 'acc': 0.608}, 'Art': {'num': 30, 'acc': 0.7}, 'Art_Theory': {'num': 30, 'acc': 0.8}, 'Design': {'num': 30, 'acc': 0.767}, 'Music': {'num': 30, 'acc': 0.167}, 'Overall-Business': {'num': 150, 'acc': 0.413}, 'Accounting': {'num': 30, 'acc': 0.467}, 'Economics': {'num': 30, 'acc': 0.4}, 'Finance': {'num': 30, 'acc': 0.4}, 'Manage': {'num': 30, 'acc': 0.4}, 'Marketing': {'num': 30, 'acc': 0.4}, 'Overall-Science': {'num': 150, 'acc': 0.38}, 'Biology': {'num': 30, 'acc': 0.6}, 'Chemistry': {'num': 30, 'acc': 0.233}, 'Geography': {'num': 30, 'acc': 0.4}, 'Math': {'num': 30, 'acc': 0.333}, 'Physics': {'num': 30, 'acc': 0.333}, 'Overall-Health and Medicine': {'num': 150, 'acc': 0.433}, 'Basic_Medical_Science': {'num': 30, 'acc': 0.5}, 'Clinical_Medicine': {'num': 30, 'acc': 0.5}, 'Diagnostics_and_Laboratory_Medicine': {'num': 30, 'acc': 0.333}, 'Pharmacy': {'num': 30, 'acc': 0.367}, 'Public_Health': {'num': 30, 'acc': 0.467}, 'Overall-Humanities and Social Science': {'num': 120, 'acc': 0.617}, 'History': {'num': 30, 'acc': 0.633}, 'Literature': {'num': 30, 'acc': 0.8}, 'Sociology': {'num': 30, 'acc': 0.567}, 'Psychology': {'num': 30, 'acc': 0.467}, 'Overall-Tech and Engineering': {'num': 210, 'acc': 0.362}, 'Agriculture': {'num': 30, 'acc': 0.567}, 'Architecture_and_Engineering': {'num': 30, 'acc': 0.267}, 'Computer_Science': {'num': 30, 'acc': 0.367}, 'Electronics': {'num': 30, 'acc': 0.3}, 'Energy_and_Power': {'num': 30, 'acc': 0.333}, 'Materials': {'num': 30, 'acc': 0.467}, 'Mechanical_Engineering': {'num': 30, 'acc': 0.233}, 'Overall': {'num': 900, 'acc': 0.452}}
   ```

</details>

### MMBench-EN & CN test

<details>
<summary>click to expand</summary>

1. Download the MMBench dataset using the following instructions:

   ```
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

2. After preparation is complete, the directory structure is:

   ```
   data
    ├── mmbench
    │   ├── CCBench_legacy.tsv
    │   ├── mmbench_dev_20230712.tsv
    │   ├── mmbench_dev_cn_20231003.tsv
    │   ├── mmbench_dev_en_20231003.tsv
    │   ├── mmbench_test_cn_20231003.tsv
    │   └── mmbench_test_en_20231003.tsv
   ```

3. Test the model with the following commands:

   We use a maximum of `6 tiles` to test the MMBench dataset.

   ```shell
   # evaluation on the test-en set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmbench-test-en --dynamic
   # evaluation on the test-cn set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmbench-test-cn --dynamic
   ```

   Submit the result to the [test server](mmbench.opencompass.org.cn). The result of MMBench is:

   ```
   # result of the test-en set
   A_Overall (test)	0.8217488789237668
   # result of the test-cn set
   A_Overall (test)	0.8195067264573991
   ```

</details>

### CCBench dev

<details>
<summary>click to expand</summary>

1. See the `MMBench-EN & CN test` part to prepare the CCBench data.

2. Test the model with the following commands:

   We use a maximum of `6 tiles` to test the CCBench dataset.

   ```shell
   # evaluation on the dev set
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 ccbench-dev --dynamic
   ```

   Submit the result to the [test server](mmbench.opencompass.org.cn). The result of CCBench is:

   ```
   A_Overall (dev)	0.7
   ```

</details>

</details>

### MMVet

<details>
<summary>click to expand</summary>

1. Download the MMVet dataset using the following instructions:

   ```
   mkdir -p data/mm-vet && cd data/mm-vet
   wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
   unzip mm-vet.zip
   wget https://huggingface.co/OpenGVLab/InternVL/raw/main/llava-mm-vet.jsonl
   cd ../..
   ```

2. After preparation is complete, the directory structure is:

   ```
   data
    ├── mm-vet
    │   ├── images
    │   └── llava-mm-vet.jsonl
   ```

3. Test the model with the following commands:

   We use a maximum of `6 tiles` to test the MMVet dataset.

   ```shell
   # evaluation on the mmvet
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mmvet --dynamic
   ```

   Submit the result to the [test server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator). The result of MMVet is:

   ```
   total
   62.7
   ```

</details>

### SEED Image

<details>
<summary>click to expand</summary>

1. Download the SEED dataset using the following instructions:

   ```
   mkdir -p data/SEED && cd data/SEED
   # 1. Follow the official instructions [Data Preparation for SEED-Bench-1](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md#data-preparation-for-seed-bench-1)
   #    to download the images and the videos. Put images under `./data/SEED/SEED-Bench-image`.
   # 2. Extract the video frame in the middle from the downloaded videos, and put them under `./data/SEED/SEED-Bench-image`.
   #    LLaVA provided the script [`extract_video_frames.py`](../internvl_chat/tools/extract_video_frames.py) modified from the official one.

   wget https://huggingface.co/OpenGVLab/InternVL/raw/main/seed.jsonl
   cd ../..
   ```

2. After preparation is complete, the directory structure is:

   ```
   data
    ├── SEED
    │   ├── SEED-Bench-image
    │   └── seed.jsonl
   ```

3. Test the model with the following commands:

   ```shell
   sh evaluate.sh pretrained/InternVL-Chat-V1-5 seed --dynamic
   ```

   The result is:

   ```
   Acc@1: 0.6999444135630906
   length: 17990
   Accuracy for each data type:
   Data type Scene Understanding: 80.37%
   Data type Instance Identity: 80.45%
   Data type Instance Location: 78.03%
   Data type Instance Attributes: 72.39%
   Data type Instances Counting: 69.19%
   Data type Spatial Relation: 59.82%
   Data type Instance Interaction: 77.32%
   Data type Visual Reasoning: 78.85%
   Data type Text Understanding: 55.81%
   Data type Action Recognition: 54.08%
   Data type Action Prediction: 44.82%
   Data type Procedure Understanding: 40.18%
   Total accuracy: 69.99%
   Image accuracy: 75.99%
   Video accuracy: 47.27%
   ```

</details>

### HallusionBench

<details>
<summary>click to expand</summary>

Please use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for the test of HallusionBench.

The command to test InternVL-Chat-V1-5 on HallusionBench using VLMEvalKit is:

```
torchrun --nproc-per-node=8 run.py --data HallusionBench --model InternVL-Chat-V1-5 --verbose
```

The result is:

```
2024-04-29 00:46:23,688 - Evaluation - INFO - Score:
2024-04-29 00:46:23,688 - Evaluation - INFO -           split       aAcc       fAcc       qAcc
0       Overall  66.771819  40.173410  40.879121
1            VD  63.620981  40.000000  34.296029
2            VS  71.944444  40.517241  51.123596
3     VD_figure  77.500000  65.853659  53.846154
4        VS_map  56.250000  18.181818  18.750000
5   VD_illusion  66.666667  41.935484  34.722222
6      VS_table  75.892857  46.428571  55.813953
7        VD_ocr  78.651685  58.139535  58.139535
8        VS_ocr  59.259259  38.461538  22.222222
9      VS_chart  81.538462  50.000000  72.368421
10     VD_video  51.176471  10.416667  13.043478
11      VD_math  56.481481  25.000000  27.777778
```

The final score reported in our technical report is the average: (66.771819 + 40.173410 + 40.879121) / 3 = 49.3

</details>

## Math Benchmark

Finally, we use a representative math dataset, MathVista, to test InternVL-Chat-V1.5.

### MathVista testmini

<details>
<summary>click to expand</summary>

1. Download the MathVista dataset using the following instructions:

   ```bash
   mkdir -p data/MathVista && cd data/MathVista
   wget https://huggingface.co/datasets/AI4Math/MathVista/raw/main/annot_testmini.json
   cd ../..
   ```

2. Test the model with the following commands:

   ```shell
   export OPENAI_API_KEY='your-openai-key'
   GPUS=8 sh evaluate.sh pretrained/InternVL-Chat-V1-5 mathvista-testmini --dynamic
   ```

   The result is:

   ```
   Correct: 535, Total: 1000, Accuracy: 53.5%
   1000
   Number of test problems: 1000

   Type: [question_type]
   [free_form]: 47.17% (217/460)
   [multi_choice]: 58.89% (318/540)

   Type: [answer_type]
   [float]: 0.00% (0/40)
   [integer]: 51.67% (216/418)
   [text]: 58.89% (318/540)
   [list]: 50.00% (1/2)

   Type: [language]
   [english]: 53.31% (499/936)
   [chinese]: 56.45% (35/62)
   [persian]: 50.00% (1/2)
   ```

</details>
