# InternViT-6B for Semantic Segmentation

This folder contains the implementation of the InternViT-6B for semantic segmentation.

Our segmentation code is developed on top of [MMSegmentation v0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0).

## 🛠️ Install

- Clone this repo:

  ```bash
  git clone https://github.com/OpenGVLab/InternVL.git
  cd InternVL/segmentation
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

- Install `flash-attn==0.2.8` :

  If you want to fully replicate my results, please install `v0.2.8`, otherwise install the latest version.

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v0.2.8
  python setup.py install
  ```

- Install other requirements:

  note: conda opencv will break torchvision as not to support GPU, so we need to install opencv using pip.

  ```bash
  conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
  pip install opencv-python
  ```

- Install `timm==0.6.11` and `mmcv==1.6.2` and `mmsegmentation==0.30.0`:

  ```bash
  pip install -U openmim
  mim install mmcv-full==1.6.2
  mim install mmsegmentation==0.30.0
  pip install timm==0.6.11
  pip install yapf==0.40.1
  ```

- Install `tensorboard`:

  ```bash
  pip install future tensorboard
  ```

## 📦 Data Preparation

Prepare the ADE20K dataset according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

## 🔥 Training

> Please note, this open-source code does not include DeepSpeed in MMSegmentation, so it currently only supports training for linear probing and head tuning, and does not support full-parameter training.

To train a linear classifier for `InternViT-6B` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py 8
# or manage jobs with slurm
GPUS=8 sh slurm_train.sh <partition> <job-name> configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py
```

## 📊 Evaluation

| type            | backbone              |  head   | mIoU |                                                   config                                                   |                                                                                                                      download                                                                                                                       |
| --------------- | --------------------- | :-----: | :--: | :--------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| few-shot (1/16) | InternViT-6B          | Linear  | 46.5 |     [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_5k_ade20k_bs16_lr4e-5_1of16.py)     |                                                             ckpt \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_5k_ade20k_bs16_lr4e-5_1of16.log)                                                              |
| few-shot (1/8)  | InternViT-6B          | Linear  | 50.0 |     [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_10k_ade20k_bs16_lr4e-5_1of8.py)     |                                                             ckpt \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_10k_ade20k_bs16_lr4e-5_1of8.log)                                                              |
| few-shot (1/4)  | InternViT-6B          | Linear  | 53.3 |     [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_20k_ade20k_bs16_lr4e-5_1of4.py)     |                                                             ckpt \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_20k_ade20k_bs16_lr4e-5_1of4.log)                                                              |
| few-shot (1/2)  | InternViT-6B          | Linear  | 55.8 |     [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_40k_ade20k_bs16_lr4e-5_1of2.py)     |                                                             ckpt \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_40k_ade20k_bs16_lr4e-5_1of2.log)                                                              |
| few-shot (1/1)  | InternViT-6B          | Linear  | 57.2 |     [config](./configs/intern_vit_6b/few_shot/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_1of1.py)     |                                                             ckpt \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_1of1.log)                                                              |
| linear probing  | InternViT-6B (frozen) | Linear  | 47.2 | [config](./configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py) |  [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.log)  |
| head tuning     | InternViT-6B (frozen) | UperNet | 54.9 |  [config](./configs/intern_vit_6b/head_tuning/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py)  | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth) \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.log) |
| full tuning     | InternViT-6B          | UperNet | 58.9 |     [config](./configs/intern_vit_6b/full_tuning/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.py)      |                                                               ckpt \| [log](https://huggingface.co/OpenGVLab/InternVL/raw/main/upernet_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5.log)                                                                |

You can download checkpoints from [here](https://huggingface.co/OpenGVLab/InternVL/tree/main) or from the table above. Then place them to `segmentation/checkpoints/`.

For example, to evaluate the `InternViT-6B` with a single GPU:

```bash
python test.py configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py checkpoints/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth --eval mIoU
```

For example, to evaluate the `InternViT-6B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py checkpoints/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth 8 --eval mIoU
```

## 🖼️ Image Demo

To infer a single/multiple image like this.
If you specify a directory instead of a single image, it will process all the images in the directory:

```
CUDA_VISIBLE_DEVICES=0 python image_demo.py \
  data/ade/ADEChallengeData2016/images/validation/ADE_val_00000591.jpg \
  configs/intern_vit_6b/linear_probing/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.py  \
  checkpoints/linear_intern_vit_6b_504_80k_ade20k_bs16_lr4e-5_frozen.pth  \
  --palette ade20k
```
