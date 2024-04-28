# How to evaluate InternVL-Chat-V1-5


In this tutorial, we will provide a detailed guide on how to replicate the results presented in the InternVL 1.5 technical report. The results are shown in the table below.

![image](https://github.com/OpenGVLab/InternVL/assets/23737120/8b62d429-c689-426a-9267-2727b6430b6e)

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

```sh
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

   ```shell
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

   ```shell
   
   ```
