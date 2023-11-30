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

- ImageNet-1K: We use the standard ImageNet dataset, you can download it from [http://image-net.org/](http://image-net.org/).
- ImageNet-A: Download it from [https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar).
- ImageNet-R: Download it from [https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar).
- ImageNetV2: Download it from [https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz](https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz).
- ImageNet-Sketch: Download it using `gdown`.
  ```shell
  # GDown is needed to download the dataset. Please install it via `pip install gdown`
  gdown --id 1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
  ```

Please prepare the ImageNet-1K, ImageNet-A, ImageNet-R, ImageNetV2, and ImageNet-Sketch datasets following the directory structure outlined below.

```bash
$ tree data
data
├── imagenet-1k
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
  --cfg configs/intern_vit_6b_1k_224.yaml
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224.yaml --launcher slurm
```

### Evaluation

| model name                  | IN-1K | IN-ReaL | IN-V2 | IN-A | IN-R | IN-Sketch |                              download                               |
| --------------------------- | :---: | :-----: | :---: | :--: | :--: | :-------: | :-----------------------------------------------------------------: |
| `intern_vit_6b_1k_224.yaml` | 88.2  |  90.4   | 80.0  | 77.5 | 89.8 |   69.1    | [ckpt](./) \| [log](./work_dirs/intern_vit_6b_1k_224/log_rank0.txt) |

<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-1K val</b> with 8 GPUs (click to expand).</summary>

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224.yaml --resume pretrained/intern_vit_6b_224px_head.pth
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224.yaml --eval \
    --resume pretrained/intern_vit_6b_224px_head.pth --launcher slurm
```

Expected results:

```
 * Acc@1 88.230 Acc@5 98.474
Accuracy of the network on the 50000 test images: 88.2%
```

</details>

<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-ReaL</b> with 1 GPU (click to expand).</summary>

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenet_real.yaml --resume pretrained/intern_vit_6b_224px_head.pth
# or manage jobs with slurm
GPUS=1 GPUS_PER_NODE=1 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenet_real.yaml --eval \
    --resume pretrained/intern_vit_6b_224px_head.pth --launcher slurm
```

Expected results:

```
* ReaL Acc@1 90.439 Acc@5 98.572 loss 0.605
ReaL Accuracy of the network on the 50000 test images: 90.4%
```

</details>

<details>
  <summary>Evaluate InternViT-6B on <b>ImageNetV2</b> with 8 GPUs (click to expand).</summary>

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenetv2.yaml --resume pretrained/intern_vit_6b_224px_head.pth
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenetv2.yaml --eval \
    --resume pretrained/intern_vit_6b_224px_head.pth --launcher slurm
```

Expected results:

```
 * Acc@1 79.960 Acc@5 95.340
Accuracy of the network on the 10000 test images: 80.0%
```

</details>

<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-A</b> with 8 GPUs (click to expand).</summary>

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenet_a.yaml --resume pretrained/intern_vit_6b_224px_head.pth
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenet_a.yaml --eval \
    --resume pretrained/intern_vit_6b_224px_head.pth --launcher slurm
```

Expected results:

```
 * Acc@1 77.479 Acc@5 92.724
Accuracy of the network on the 7500 test images: 77.5%
```

</details>

<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-R</b> with 8 GPUs (click to expand).</summary>

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenet_r.yaml --resume pretrained/intern_vit_6b_224px_head.pth
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenet_r.yaml --eval \
    --resume pretrained/intern_vit_6b_224px_head.pth --launcher slurm
```

Expected results:

```
 * Acc@1 89.783 Acc@5 97.023
Accuracy of the network on the 30000 test images: 89.8%
```

</details>

<details>
  <summary>Evaluate InternViT-6B on <b>ImageNet-Sketch</b> with 8 GPUs (click to expand).</summary>

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenet_sketch.yaml --resume pretrained/intern_vit_6b_224px_head.pth
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenet_sketch.yaml --eval \
    --resume pretrained/intern_vit_6b_224px_head.pth --launcher slurm
```

Expected results:

```
 * Acc@1 69.102 Acc@5 88.333
Accuracy of the network on the 50889 test images: 69.1%
```

</details>
