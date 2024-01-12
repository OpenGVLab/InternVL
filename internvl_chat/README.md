# InternVL-Chat

This folder contains the implementation of the InternVL-Chat.

***Note: This folder is still under construction.***

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
  ```

## 📦 Data Preparation

Coming Soon

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

## 📊 Evaluation

| model         | QLLaMA | LLM          | res | COCO  | Flickr | NoCaps | VQAv2 | GQA  | VizWiz | TextVQA | MME    | POPE | Download |
| ------------- | ------ | ------------ | --- | ----- | ------ | ------ | ----- | ---- | ------ | ------- | ------ | ---- | -------- |
| InternVL-Chat | ✔️     | frozen V-7B  | 224 | 141.4 | 89.7   | 120.5  | 72.3  | 57.7 | 44.5   | 42.1    | 1298.5 | 85.2 | TODO     |
| InternVL-Chat | ✔️     | frozen V-13B | 224 | 142.4 | 89.9   | 123.1  | 71.7  | 59.5 | 54.0   | 49.1    | 1317.2 | 85.4 | TODO     |
| InternVL-Chat | ✔      | V-13B        | 336 | 146.2 | 92.2   | 126.2  | 81.2  | 66.6 | 58.5   | 61.5    | 1586.4 | 87.6 | TODO     |
