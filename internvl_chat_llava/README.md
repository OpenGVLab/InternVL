# InternVL for Multimodal Dialogue using LLaVA

This folder contains the implementation of the InternVL-Chat.

We mainly use the LLaVA codebase to evaluate InternVL in creating multimodal dialogue systems. Thanks for this great work.

We have retained the original documentation of LLaVA 1.5 as a more detailed manual. In most cases, you will only need to refer to the new documentation that we have added.

> Note: To unify the environment across different tasks, we have made some compatibility modifications to the LLaVA code, allowing it to support `transformers==4.36.2` (originally locked at 4.31.0). Please note that `transformers==4.36.2` should be installed.

## 🛠️ Installation

See [INSTALLATION.md](../INSTALLATION.md)

In addition, using this codebase requires executing the following steps:

- Install other requirements:
  
  ```bash
  pip install --upgrade pip  # enable PEP 660 support
  pip install -e .
  ```

## 📦 Model Preparation

| model name         | type        | download                                                          |  size   |
| ------------------ | ----------- | ----------------------------------------------------------------- | :-----: |
| InternViT-6B-224px | huggingface | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-224px) | 12 GB |
| InternViT-6B-448px | huggingface | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px) | 12 GB |
| Vicuna-7B-v1.5     | huggingface | 🤗 [HF link](https://huggingface.co/lmsys/vicuna-7b-v1.5)         | 13.5 GB |
| Vicuna-13B-v1.5    | huggingface | 🤗 [HF link](https://huggingface.co/lmsys/vicuna-13b-v1.5)        | 26.1 GB |

Please download the above model weights and place them in the `pretrained/` folder.

```sh
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternViT-6B-224px --local-dir intern_vit_6b_224px
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternViT-6B-448px --local-dir intern_vit_6b_448px
huggingface-cli download --resume-download --local-dir-use-symlinks False lmsys/vicuna-13b-v1.5 --local-dir vicuna-13b-v1.5
huggingface-cli download --resume-download --local-dir-use-symlinks False lmsys/vicuna-7b-v1.5 --local-dir vicuna-7b-v1.5
```

The directory structure is:

```sh
pretrained
│── intern_vit_6b_224px/
│── intern_vit_6b_448px/
│── vicuna-13b-v1.5/
└── vicuna-7b-v1.5/
```

## 🔥 Training

- InternViT-6B-224px + Vicuna-7B:

```shell
# pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts_internvl/pretrain_internvit6b_224to336_vicuna7b.sh
# finetune
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts_internvl/finetune_internvit6b_224to336_vicuna7b.sh
```

- InternViT-6B-224px + Vicuna-13B:

```shell
# pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts_internvl/pretrain_internvit6b_224to336_vicuna13b.sh
# finetune
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts_internvl/finetune_internvit6b_224to336_vicuna13b.sh
```

- InternViT-6B-448px + Vicuna-7B:

```shell
# pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts_internvl/pretrain_internvit6b_448_vicuna7b.sh
# finetune
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts_internvl/finetune_internvit6b_448_vicuna7b.sh
```

- InternViT-6B-448px + Vicuna-13B:

```shell
# pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts_internvl/pretrain_internvit6b_448_vicuna13b.sh
# finetune
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts_internvl/finetune_internvit6b_448_vicuna13b.sh
```


## 🤗 Model Zoo

| method        | vision encoder | LLM   | res. | VQAv2 | GQA  | VizWiz | SQA | TextVQA | POPE | MME    | MMB   | MMB<sub>CN</sub> | MMVet | Download |                                                           
| ------------- |:--------------:|:-----:|:----:|:-----:|:----:|:------:|:-----:|:-------:|:----:|:------:|:-----:|:------:|:------:| :-----------------------------------------------------------------------------------:|
| LLaVA-1.5     | CLIP-L-336px   | V-7B  | 336  | 78.5  | 62.0 | 50.0   | 66.8  | 58.2    | 85.9 | 1510.7 | 64.3  | 58.3   | 30.5   | 🤗 [HF link](https://huggingface.co/liuhaotian/llava-v1.5-7b)                        |
| LLaVA-1.5     | CLIP-L-336px   | V-13B | 336  | 80.0  | 63.3 | 53.6   | 71.6  | 61.3    | 85.9 | 1531.3 | 67.7  | 63.6   | 35.4   | 🤗 [HF link](https://huggingface.co/liuhaotian/llava-v1.5-13b)                       |
| InternVL-Chat | IViT-6B-224px  | V-7B  | 336  | 79.3  | 62.9 | 52.5   | 66.2  | 57.0    | 86.4 | 1525.1 | 64.6  | 57.6   | 31.2   | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B)        |
| InternVL-Chat | IViT-6B-224px  | V-13B | 336  | 80.2  | 63.9 | 54.6   | 70.1  | 58.7    | 87.1 | 1546.9 | 66.5  | 61.9   | 33.7   | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B)       |
| InternVL-Chat | IViT-6B-448px  | V-13B | 448  | 82.0  | 64.1 | 60.1   | 71.6  | 64.8    | 87.2 | 1579.0 | 68.2  | 64.0   | 36.7   | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B-448px) |

Please download the above model weights and place them in the `pretrained/` folder.

```shell
cd pretrained/
# pip install -U huggingface_hub
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B --local-dir InternVL-Chat-ViT-6B-Vicuna-7B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B --local-dir InternVL-Chat-ViT-6B-Vicuna-13B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B-448px --local-dir InternVL-Chat-ViT-6B-Vicuna-13B-448px

```

The directory structure is:

```
pretrained
│── intern_vit_6b_224px/
│── intern_vit_6b_448px/
│── vicuna-13b-v1.5/
│── vicuna-7b-v1.5/
│── InternVL-Chat-ViT-6B-Vicuna-7B/
│── InternVL-Chat-ViT-6B-Vicuna-13B/
└── InternVL-Chat-ViT-6B-Vicuna-13B-448px/
```

## 🖥️ Demo

The method for deploying the demo is consistent with LLaVA-1.5. You only need to change the model path. The specific steps are as follows:

**Launch a controller**
   
 ```shell
 python -m llava.serve.controller --host 0.0.0.0 --port 10000
 ```

**Launch a gradio web server**
   
 ```shell
 python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
 ```

**Launch a model worker**
   
 ```shell
 # OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B
 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./pretrained/InternVL-Chat-ViT-6B-Vicuna-7B
 # OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B
 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40001 --worker http://localhost:40001 --model-path ./pretrained/InternVL-Chat-ViT-6B-Vicuna-13B
 ```

For more details on deploying the demo, please refer to [here](#gradio-web-ui).

## 💡Testing

The method for testing the model remains the same as LLaVA-1.5; you just need to change the path of the script. Our scripts are located in `scripts_internvl/`.

For example, testing MME using a single GPU:

```shell
sh scripts_internvl/eval/mme.sh pretrained/InternVL-Chat-ViT-6B-Vicuna-7B/
```

-------------

# 🌋 LLaVA: Large Language and Vision Assistant

*Visual instruction tuning towards large language and vision models with GPT-4 level capabilities.*

[[Project Page](https://llava-vl.github.io/)] [[Demo](https://llava.hliu.cc/)]  [[Data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)] [[Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)]

🤝Community Contributions: [[llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3436)] [[Colab](https://github.com/camenduru/LLaVA-colab)] [[🤗Space](https://huggingface.co/spaces/badayvedat/LLaVA)]

**Improved Baselines with Visual Instruction Tuning** [[Paper](https://arxiv.org/abs/2310.03744)] <br>
[Haotian Liu](https://hliu.cc), [Chunyuan Li](https://chunyuan.li/), [Yuheng Li](https://yuheng-li.github.io/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)

**Visual Instruction Tuning** (NeurIPS 2023, **Oral**) [[Paper](https://arxiv.org/abs/2304.08485)]<br>
[Haotian Liu*](https://hliu.cc), [Chunyuan Li*](https://chunyuan.li/), [Qingyang Wu](https://scholar.google.ca/citations?user=HDiw-TsAAAAJ&hl=en/), [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/) (*Equal Contribution)

<p align="center">
    <a href="https://llava.hliu.cc/"><img src="images/llava_logo.png" width="50%"></a> <br>
    Generated by <a href="https://gligen.github.io/">GLIGEN</a> via "a cute lava llama with glasses" and box prompt
</p>

## Release

- [10/12] 🔥 Check out the Korean LLaVA (Ko-LLaVA), created by ETRI, who has generously supported our research! [[🤗 Demo](https://huggingface.co/spaces/etri-vilab/Ko-LLaVA)]

- [10/12] LLaVA is now supported in [llama.cpp](https://github.com/ggerganov/llama.cpp/pull/3436) with 4-bit / 5-bit quantization support!

- [10/11] The training data and scripts of LLaVA-1.5 are released [here](https://github.com/haotian-liu/LLaVA#train), and evaluation scripts are released [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)!

- [10/5] 🔥 LLaVA-1.5 is out! Achieving SoTA on 11 benchmarks, with just simple modifications to the original LLaVA, utilizes all public data, completes training in ~1 day on a single 8-A100 node, and surpasses methods like Qwen-VL-Chat that use billion-scale data. Check out the [technical report](https://arxiv.org/abs/2310.03744), and explore the [demo](https://llava.hliu.cc/)! Models are available in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

- [9/26] LLaVA is improved with reinforcement learning from human feedback (RLHF) to improve fact grounding and reduce hallucination. Check out the new SFT and RLHF checkpoints at project [[LLavA-RLHF]](https://llava-rlhf.github.io/)

- [9/22] [LLaVA](https://arxiv.org/abs/2304.08485) is accepted by NeurIPS 2023 as **oral presentation**, and [LLaVA-Med](https://arxiv.org/abs/2306.00890) is accepted by NeurIPS 2023 Datasets and Benchmarks Track as **spotlight presentation**.

- [9/20] We summarize our empirical study of training 33B and 65B LLaVA models in a [note](https://arxiv.org/abs/2309.09958). Further, if you are interested in the comprehensive review, evolution and trend of multimodal foundation models, please check out our recent survey paper [``Multimodal Foundation Models: From Specialists to General-Purpose Assistants''.](https://arxiv.org/abs/2309.10020)
  
  <p align="center">
  <img src="https://github.com/Computer-Vision-in-the-Wild/CVinW_Readings/blob/main/images/mfm_evolution.jpeg?raw=true" width=50%/>
  </p>

- [7/19] 🔥 We release a major upgrade, including support for LLaMA-2, LoRA training, 4-/8-bit inference, higher resolution (336x336), and a lot more. We release [LLaVA Bench](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_Bench.md) for benchmarking open-ended visual chat with results from Bard and Bing-Chat. We also support and verify training with RTX 3090 and RTX A6000. Check out [LLaVA-from-LLaMA-2](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_from_LLaMA2.md), and our [model zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)!

- [6/26] [CVPR 2023 Tutorial](https://vlp-tutorial.github.io/) on **Large Multimodal Models: Towards Building and Surpassing Multimodal GPT-4**!  Please check out [[Slides](https://datarelease.blob.core.windows.net/tutorial/vision_foundation_models_2023/slides/Chunyuan_cvpr2023_tutorial_lmm.pdf)] [[Notes](https://arxiv.org/abs/2306.14895)] [[YouTube](https://youtu.be/mkI7EPD1vp8)] [[Bilibli](https://www.bilibili.com/video/BV1Ng4y1T7v3/)].

- [6/11] We released the preview for the most requested feature: DeepSpeed and LoRA support!  Please see documentations [here](./docs/LoRA.md).

- [6/1] We released **LLaVA-Med: Large Language and Vision Assistant for Biomedicine**, a step towards building biomedical domain large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2306.00890) and [page](https://github.com/microsoft/LLaVA-Med).

- [5/6] We are releasing [LLaVA-Lighting-MPT-7B-preview](https://huggingface.co/liuhaotian/LLaVA-Lightning-MPT-7B-preview), based on MPT-7B-Chat!  See [here](#LLaVA-MPT-7b) for more details.

- [5/2] 🔥 We are releasing LLaVA-Lighting!  Train a lite, multimodal GPT-4 with just $40 in 3 hours!  See [here](#train-llava-lightning) for more details.

- [4/27] Thanks to the community effort, LLaVA-13B with 4-bit quantization allows you to run on a GPU with as few as 12GB VRAM!  Try it out [here](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/llava).

- [4/17] 🔥 We released **LLaVA: Large Language and Vision Assistant**. We propose visual instruction tuning, towards building large language and vision models with GPT-4 level capabilities.  Checkout the [paper](https://arxiv.org/abs/2304.08485) and [demo](https://llava.hliu.cc/).

<!-- <a href="https://llava.hliu.cc/"><img src="assets/demo.gif" width="70%"></a> -->

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## Contents

- [Install](#install)
- [LLaVA Weights](#llava-weights)
- [Demo](#Demo)
- [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)
- [Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to LLaVA folder
   
   ```bash
   git clone https://github.com/haotian-liu/LLaVA.git
   cd LLaVA
   ```

2. Install Package
   
   ```Shell
   conda create -n llava python=3.10 -y
   conda activate llava
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .
   ```

3. Install additional packages for training cases
   
   ```
   pip install ninja
   pip install flash-attn --no-build-isolation
   ```

### Upgrade to latest code base

```Shell
git pull
pip uninstall transformers
pip install -e .
```

## LLaVA Weights

Please check out our [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) for all public LLaVA checkpoints, and the instructions of how to use the weights.

## Demo

To run our demo, you need to prepare LLaVA checkpoints locally.  Please follow the instructions [here](#llava-weights) to download the checkpoints.

### Gradio Web UI

To launch a Gradio demo locally, please run the following commands one by one. If you plan to launch multiple model workers to compare between different checkpoints, you only need to launch the controller and the web server *ONCE*.

#### Launch a controller

```Shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a gradio web server.

```Shell
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

You just launched the Gradio web interface. Now, you can open the web interface with the URL printed on the screen. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker.

#### Launch a model worker

This is the actual *worker* that performs the inference on the GPU.  Each worker is responsible for a single model specified in `--model-path`.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...".  Now, refresh your Gradio web UI, and you will see the model you just launched in the model list.

You can launch as many workers as you want, and compare between different model checkpoints in the same Gradio interface. Please keep the `--controller` the same, and modify the `--port` and `--worker` to a different port number for each worker.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port <different from 40000, say 40001> --worker http://localhost:<change accordingly, i.e. 40001> --model-path <ckpt2>
```

If you are using an Apple device with an M1 or M2 chip, you can specify the mps device by using the `--device` flag: `--device mps`.

#### Launch a model worker (Multiple GPUs, when GPU VRAM <= 24GB)

If the VRAM of your GPU is less than 24GB (e.g., RTX 3090, RTX 4090, etc.), you may try running it with multiple GPUs. Our latest code base will automatically try to use multiple GPUs if you have more than one GPU. You can specify which GPUs to use with `CUDA_VISIBLE_DEVICES`. Below is an example of running with the first two GPUs.

```Shell
CUDA_VISIBLE_DEVICES=0,1 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b
```

#### Launch a model worker (4-bit, 8-bit inference, quantized)

You can launch the model worker with quantized bits (4-bit, 8-bit), which allows you to run the inference with reduced GPU memory footprint, potentially allowing you to run on a GPU with as few as 12GB VRAM. Note that inference with quantized bits may not be as accurate as the full-precision model. Simply append `--load-4bit` or `--load-8bit` to the **model worker** command that you are executing. Below is an example of running with 4-bit quantization.

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b --load-4bit
```

#### Launch a model worker (LoRA weights, unmerged)

You can launch the model worker with LoRA weights, without merging them with the base checkpoint, to save disk space. There will be additional loading time, while the inference speed is the same as the merged checkpoints. Unmerged LoRA checkpoints do not have `lora-merge` in the model name, and are usually much smaller (less than 1GB) than the merged checkpoints (13G for 7B, and 25G for 13B).

To load unmerged LoRA weights, you simply need to pass an additional argument `--model-base`, which is the base LLM that is used to train the LoRA weights. You can check the base LLM of each LoRA weights in the [model zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md).

```Shell
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3 --model-base lmsys/vicuna-13b-v1.3
```

### CLI Inference

Chat about images using LLaVA without the need of Gradio interface. It also supports multiple GPUs, 4-bit and 8-bit quantized inference. With 4-bit quantization, for our LLaVA-1.5-7B, it uses less than 8GB VRAM on a single GPU.

```Shell
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
```

<img src="images/demo_cli.gif" width="70%">

## Train

*Below is the latest training configuration for LLaVA v1.5. For legacy models, please refer to README of [this](https://github.com/haotian-liu/LLaVA/tree/v1.0.1) version for now. We'll add them in a separate doc later.*

LLaVA training consists of two stages: (1) feature alignment stage: use our 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following data, plus around 515K VQA data from academic-oriented tasks, to teach the model to follow multimodal instructions.

LLaVA is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters

We use a similar set of hyperparameters as Vicuna in finetuning.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| -------------- | -----------------:| -------------:| ------:| ----------:| ------------:|
| LLaVA-v1.5-13B | 256               | 1e-3          | 1      | 2048       | 0            |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| -------------- | -----------------:| -------------:| ------:| ----------:| ------------:|
| LLaVA-v1.5-13B | 128               | 2e-5          | 1      | 2048       | 0            |

### Download Vicuna checkpoints (automatically)

Our base model Vicuna v1.5, which is an instruction-tuned chatbot, will be downloaded automatically when you run our provided training scripts. No action is needed.

### Pretrain (feature alignment)

Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions we use in the paper [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Pretrain takes around 5.5 hours for LLaVA-v1.5-13B on 8x A100 (80G), due to the increased resolution to 336px. It takes around 3.5 hours for LLaVA-v1.5-7B.

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh).

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.

### Visual Instruction Tuning

1. Prepare data

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

2. Start training!

You may download our pretrained projectors in [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md). It is not recommended to use legacy projectors, as they may be trained with a different version of the codebase, and if any option is off, the model will not function/train as we expected.

Visual instruction tuning takes around 20 hours for LLaVA-v1.5-13B on 8x A100 (80G), due to the increased resolution to 336px. It takes around 10 hours for LLaVA-v1.5-7B on 8x A100 (40G).

Training script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune.sh).

New options to note:

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--image_aspect_ratio pad`: this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
- `--group_by_modality_length True`: this should only be used when your instruction tuning dataset contains both language (e.g. ShareGPT) and multimodal (e.g. LLaVA-Instruct). It makes the training sampler only sample a single modality (either image or language) during training, which we observe to speed up training by ~25%, and does not affect the final outcome.

## Evaluation

In LLaVA-1.5, we evaluate models on a diverse set of 12 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

### GPT-assisted Evaluation

Our GPT-assisted evaluation pipeline for multimodal modeling is provided for a comprehensive understanding of the capabilities of vision-language models.  Please see our paper for more details.

1. Generate LLaVA responses

```Shell
python model_vqa.py \
    --model-path ./checkpoints/LLaVA-13B-v0 \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /path/to/coco2014_val \
    --answers-file \
    /path/to/answer-file-our.jsonl
```

2. Evaluate the generated responses.  In our case, [`answer-file-ref.jsonl`](./playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl) is the response generated by text-only GPT-4 (0314), with the context captions/boxes provided.

```Shell
OPENAI_API_KEY="sk-***********************************" python llava/eval/eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    /path/to/answer-file-ref.jsonl \
    /path/to/answer-file-our.jsonl \
    --rule llava/eval/table/rule.json \
    --output /path/to/review.json
```

3. Summarize the evaluation results

```Shell
python summarize_gpt_review.py
```

## Citation

If you find LLaVA useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={arXiv:2304.08485},
      year={2023},
}
```

## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!

## Related Projects

- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)

For future project ideas, please check out:

- [SEEM: Segment Everything Everywhere All at Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to detect, segment, and generate anything by marrying [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).
