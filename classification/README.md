# InternViT-6B for Image Classification

This folder contains the implementation of the InternViT-6B for image classification.

<!-- TOC -->

- [Install](#install)
- [Data Preparation](#data-preparation)
- [Linear Probing on ImageNet-1K](#linear-probing-on-imagenet-1k)
- [Evaluation](#evaluation)

<!-- TOC -->

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVL/classification
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

- Install `timm==0.6.11` and `mmcv==1.7.0`:

```bash
pip install -U openmim
pip install timm==0.6.11
mim install mmcv==1.7.0
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
```

### Data Preparation

Please prepare the ImageNet-1K, ImageNet-A, ImageNet-R, ImageNet-Sketch, and ImageNetV2 datasets following the directory structure outlined below.

```bash
$ tree data
data
├── imagenet-1k
│         ├── meta
│         ├── test
│         ├── train
          │    ├── n01498041
          │    └── ...
│         └── val
│              ├── ILSVRC2012_val_00000001.JPEG
│              └── ...
├── imagenet-a
│         ├── n01498041
│         └── ...
├── imagenet-r
│         ├── n01443537
│         └── ...
├── imagenet-sketch
│         ├── n01440764
│         └── ...
└── imagenetv2
    └── ImageNetV2-matched-frequency
```

### Linear Probing on ImageNet-1K


To train a linear classifier for `InternViT-6b` on ImageNet with 8 GPUs, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py \
  --cfg configs/intern_vit_6b_1k_224.yaml --data-path ./data/imagenet-1k
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224.yaml --launcher slurm
```

### Evaluation

| model name | IN-1K | IN-ReaL | IN-V2 | IN-A | IN-R | IN-Sketch |
|--------|:------------------:|:------:|:------:| :------:|:------:|:------:|
| `intern_vit_6b_1k_224.yaml` | 88.2 | 90.4 | 80.0 | 77.4 | 89.8 | 69.0 |

<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-1K val</b> with 8 GPUs (click to expand).</summary>


```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224.yaml --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-1k
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224.yaml --eval \
    --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-1k --launcher slurm
```

Expected results:

```
 * Acc@1 88.244 Acc@5 98.470
Accuracy of the network on the 50000 test images: 88.2%
```

</details>



<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-ReaL val</b> with 1 GPU (click to expand).</summary>


```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenet_real.yaml --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-1k
# or manage jobs with slurm
GPUS=1 GPUS_PER_NODE=1 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenet_real.yaml --eval \
    --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-1k --launcher slurm
```

Expected results:

```
* ReaL Acc@1 90.377 Acc@5 98.557 loss 0.596
ReaL Accuracy of the network on the 50000 test images: 90.4%
```

</details>



<details>
  <summary>Evaluate InternViT-6B on <b>ImageNetV2</b> with 8 GPUs (click to expand).</summary>


```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenetv2.yaml --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenetv2
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenetv2.yaml --eval \
    --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenetv2 --launcher slurm
```

Expected results:

```
 * Acc@1 79.960 Acc@5 95.300
Accuracy of the network on the 10000 test images: 80.0%
```

</details>

<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-A</b> with 8 GPUs (click to expand).</summary>


```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenet_a.yaml --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-a
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenet_a.yaml --eval \
    --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-a --launcher slurm
```

Expected results:

```
 * Acc@1 77.400 Acc@5 92.720
Accuracy of the network on the 7500 test images: 77.4%
```

</details>


<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-R</b> with 8 GPUs (click to expand).</summary>


```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenet_r.yaml --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-r
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenet_r.yaml --eval \
    --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-r --launcher slurm
```

Expected results:

```

```

</details>



<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-Sketch</b> with 8 GPUs (click to expand).</summary>


```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenet_sketch.yaml --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-sketch
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenet_sketch.yaml --eval \
    --resume intern_vit_6b_224px_head.pth --data-path ./data/imagenet-sketch --launcher slurm
```

Expected results:

```

```

</details>
