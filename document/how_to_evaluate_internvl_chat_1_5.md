# How to evaluate InternVL-Chat-V1-5


In this tutorial, we will provide a detailed guide on how to replicate the results presented in the InternVL 1.5 technical report. 

The results are shown in the table below.

![image](https://github.com/OpenGVLab/InternVL/assets/23737120/8b62d429-c689-426a-9267-2727b6430b6e)

> Note that if you are aiming for an exact replication, please use this code repository and follow the testing methods outlined below; otherwise, using the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) would be more convenient.

## Model Preparation

| model name         | type  | download                                                          |  #param |
| ------------------ | ----- | ----------------------------------------------------------------- | :-----: |
| InternVL-Chat-V1-5 | MLLM  | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) | 25.5B   |

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

Our tests will be divided into two parts. First, we will focus on OCR-related datasets, including DocVQA, ChartQA, InfoVQA, TextVQA, and OCRBench. Next, let's proceed to test each dataset one by one.

### DocVQA


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

   We use a maximum of 18 tiles to test the DocVQA dataset:
     
   ```shell
   # evaluation on the val set
   sh evaluate.sh release/InternVL-Chat-V1-5 vqa-docvqa-val --dynamic --max-num 18
   # evaluation on the test set
   sh evaluate.sh release/InternVL-Chat-V1-5 vqa-docvqa-test --dynamic --max-num 18
   ```

   The result of the validation set is:

   ```
   Overall ANLS: 0.9049
   ```

   For the test set, the test results need to be submitted to the [testing server](https://rrc.cvc.uab.es/?ch=17&com=tasks).

### ChartQA

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
    │   ├── test_augmented.jsonl
    │   ├── test_human.jsonl
    │   ├── train_augmented.jsonl
    │   └── train_human.jsonl
   ```

3. Test the model with the following commands:

   We use a maximum of 12 tiles to test the ChartQA dataset:
     
   ```shell
   # evaluation on the test set
   sh evaluate.sh release/InternVL-Chat-V1-5 vqa-chartqa-test --dynamic --max-num 12
   ```

   The result of the test set is:

   ```
   ['chartqa_test_human', {'relaxed_accuracy': 0.736}]
   ['chartqa_test_augmented', {'relaxed_accuracy': 0.9408}]
   # the average score = (73.6 + 94.08) / 2 = 83.8
   ```

### InfoVQA

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

3. After preparation is complete, the directory structure is:

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

4. Test the model with the following commands:

   We use a maximum of 24 tiles to test the InfoVQA dataset:
     
   ```shell
   # evaluation on the val set
   sh evaluate.sh release/InternVL-Chat-V1-5 vqa-infovqa-val --dynamic --max-num 24
   # evaluation on the test set
   sh evaluate.sh release/InternVL-Chat-V1-5 vqa-infovqa-test --dynamic --max-num 24
   ```

   The result of the val set is:

   ```
   Overall ANLS: 0.7235
   ```

   For the test set, the test results need to be submitted to the [testing server](https://rrc.cvc.uab.es/?ch=17&com=tasks).

### OCRBench

Please use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for the test of OCRBench.
