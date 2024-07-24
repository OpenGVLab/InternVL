# InternViT-6B for Image Classification

This folder contains the implementation of the InternViT-6B for image classification, which corresponds to Section 4.2.1 of our [InternVL 1.0 paper](https://arxiv.org/pdf/2312.14238).
The codebase for this part is derived from [InternImage](https://github.com/OpenGVLab/InternImage), with some code references to [EVA](https://github.com/baaivision/EVA/tree/master) and [DINOv2](https://github.com/facebookresearch/dinov2). Thanks for their great work.

In this part, we validate the visual perception capabilities of InternViT-6B, the most core component of InternVL 1.0.
We evaluate the quality of visual representation produced by InternViT-6B using the ImageNet-1K dataset. Following common practices, we adopt the linear probing evaluation, i.e. training a linear classifier while keeping the backbone frozen. In addition to the ImageNet-1K validation set,
we also report performance metrics on several ImageNet variants, to benchmark the domain generalization capability.

InternViT-6B follows the structure of vanilla ViT, and its hyperparameters are listed in the table below.

<img width="558" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/e6bb0151-ab2f-4436-982f-6c68c5a69bc4">

## 🛠️ Installation

Follow the [installation guide](../INSTALLATION.md) to perform installations.

## 📦 Data Preparation

> Please prepare the dataset according to your needs.

- `ImageNet-1K`: We use the standard ImageNet dataset, you can download it from [http://image-net.org/](http://image-net.org/).

- `ImageNet-A`: Download it from [https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar).

- `ImageNet-R`: Download it from [https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar).

- `ImageNetV2`: Download it from [https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz](https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz).

- `ImageNet-Sketch`: Download it using `gdown`.

  ```shell
  # GDown is needed to download the dataset.
  # Please install it via `pip install gdown`
  gdown --id 1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
  ```

First, please prepare the `ImageNet-1K`, `ImageNet-A`, `ImageNet-R`, `ImageNetV2`, and `ImageNet-Sketch` datasets following the directory structure outlined below.

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

Then, unzip the `train.txt.zip` and `val.txt.zip` in `meta_data/`.

```shell
cd meta_data/
unzip train.txt.zip
unzip val.txt.zip
```

## 📦 Model Preparation

| model name                   | type    | download                                                                                       |  size   |
| ---------------------------- | ------- | ---------------------------------------------------------------------------------------------- | :-----: |
| intern_vit_6b_224px.pth      | pytorch | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/intern_vit_6b_224px.pth)      |  12 GB  |
| intern_vit_6b_224px_head.pth | pytorch | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/intern_vit_6b_224px_head.pth) | 25.7 MB |

Please download the above model weights and place them in the `pretrained/` folder.

```sh
cd pretrained
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_224px.pth
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_224px_head.pth
```

The directory structure is:

```sh
pretrained
├── intern_vit_6b_224px_head.pth
└── intern_vit_6b_224px.pth
```

## 🔍 Linear Probing on ImageNet-1K

> **Warning**: Please install `apex` before training (see [installation guide](../INSTALLATION.md#additional-instructions) for details).

To train a linear classifier for `InternViT-6B` on ImageNet with 8 GPUs, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --cfg configs/intern_vit_6b_1k_224.yaml
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224.yaml --launcher slurm
```

Note, it is normal for the following information to appear during training and it can be safely ignored:

> \_IncompatibleKeys(missing_keys=\[\], unexpected_keys=\['clip_projector.norm1_q.weight', 'clip_projector.norm1_q.bias', 'clip_projector.norm1_k.weight', 'clip_projector.norm1_k.bias', 'clip_projector.norm1_v.weight', 'clip_projector.norm1_v.bias', 'clip_projector.cross_attn.q_bias', 'clip_projector.cross_attn.k_bias', 'clip_projector.cross_attn.v_bias', 'clip_projector.cross_attn.q.weight', 'clip_projector.cross_attn.k.weight', 'clip_projector.cross_attn.v.weight', 'clip_projector.cross_attn.proj.weight', 'clip_projector.cross_attn.proj.bias'\])

## 📊 Evaluation

> **Warning**: Please install `apex` before evaluation (see [installation guide](../INSTALLATION.md#additional-instructions) for details).

| model name                                                     | IN-1K | IN-ReaL | IN-V2 | IN-A | IN-R | IN-Sketch |                                                                       download                                                                       |
| -------------------------------------------------------------- | :---: | :-----: | :---: | :--: | :--: | :-------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: |
| [intern_vit_6b_1k_224.yaml](configs/intern_vit_6b_1k_224.yaml) | 88.2  |  90.4   | 79.9  | 77.5 | 89.8 |   69.1    | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_224px_head.pth) \| [log](./work_dirs/intern_vit_6b_1k_224/log_rank0.txt) |

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

**Note: ImageNet-ReaL now only supports single-GPU testing.**

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
    --cfg configs/intern_vit_6b_1k_224_test_imagenet_real.yaml --resume pretrained/intern_vit_6b_224px_head.pth
# or manage jobs with slurm
GPUS=1 GPUS_PER_NODE=1 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224_test_imagenet_real.yaml --eval \
    --resume pretrained/intern_vit_6b_224px_head.pth --launcher slurm
```

Expected results:

```
* ReaL Acc@1 90.437 Acc@5 98.567 loss 0.605
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
 * Acc@1 79.940 Acc@5 95.340
Accuracy of the network on the 10000 test images: 79.9%
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
 * Acc@1 77.479 Acc@5 92.737
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
 * Acc@1 89.777 Acc@5 97.023
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
 * Acc@1 69.117 Acc@5 88.341
Accuracy of the network on the 50889 test images: 69.1%
```

</details>
