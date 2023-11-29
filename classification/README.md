# InternViT-6B for Image Classification

This folder contains the implementation of the InternViT-6B for image classification.

<!-- TOC -->
* [Install](#install)
* [Data Preparation](#data-preparation)
* [Evaluation](#evaluation)
* [Training from Scratch on ImageNet-1K](#training-from-scratch-on-imagenet-1k)
* [Manage Jobs with Slurm.](#manage-jobs-with-slurm)
* [Training with Deepspeed](#training-with-deepspeed)
* [Extracting Intermediate Features](#extracting-intermediate-features)
* [Export](#export)
<!-- TOC -->

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVeL/classification
```

- Create a conda virtual environment and activate it:

```bash
conda create -n internvl python=3.9 -y
conda activate internvl
```

- Install `PyTorch>=2.0` and `torchvision>=0.15.2` with `CUDA>=11.6`:

For examples, to install torch==2.0.1 with CUDA==11.8:
```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# or
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

- Install `timm==0.6.11` and `mmcv-full==1.7.0`:

```bash
pip install -U openmim
pip install timm==0.6.11
mim install mmcv==1.7.0
```

- Install 'apex':

```bash
git clone https://github.com/NVIDIA/apex.git
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

### Data Preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...

  ```


### Evaluation

To evaluate a pretrained `InternViT-6B` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path>
```

For example, to evaluate the `InternImage-B` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/internimage_b_1k_224.yaml --resume internimage_b_1k_224.pth --data-path <imagenet-path>
```

### Linear Evaluation on ImageNet-1K

To train an `InternImage` on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

### Manage Jobs with Slurm.

For example, to train `InternImage` with 8 GPU on a single node for 300 epochs, run:

`InternViT-6B`:

```bash
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_t_1k_224.yaml --resume internimage_t_1k_224.pth --eval
```


### Export

To export `InternViT-6B` from PyTorch to ONNX, run:
```shell
python export.py --model_name intern_vit_6b_1k_224_cls_patch_sgd_lr0.1 --ckpt_dir /path/to/ckpt/dir --onnx
```

To export `InternViT-6B` from PyTorch to TensorRT, run:
```shell
python export.py --model_name intern_vit_6b_1k_224_cls_patch_sgd_lr0.1 --ckpt_dir /path/to/ckpt/dir --trt
```
