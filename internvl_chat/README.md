# InternVL-Chat

This folder contains the implementation of the InternVL-Chat.

***Note: This folder is still under construction.***

## 🛠️ Install

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
  ```

## 📦 Data Preparation

Coming Soon

## 📦 Model Preparation

| model name   | type        | download                                                          |  size   |
| ------------ | ----------- | ----------------------------------------------------------------- | :-----: |
| InternVL-C/G | huggingface | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-14B-224px) | 27.7 GB |

Please download the above model weights and place them in the `pretrained/` folder.

You can download either the PyTorch version or the Hugging Face version based on your needs.

The directory structure is:

```sh
pretrained
└── internvl_14b_224px/

```

## 🔥 Supervised Fine-tuning

Coming Soon

## 📊 Evaluation

Coming Soon
