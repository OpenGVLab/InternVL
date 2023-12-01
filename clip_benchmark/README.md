# InternVL for Image Classification & Image-Text Retrieval

This folder contains the implementation of InternVL for image classification and image-text retrieval.

## 🛠️ Install

- Clone this repository:

  ```bash
  git clone https://github.com/OpenGVLab/InternVL.git
  cd InternVL/clip_benchmark
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

- Install `timm==0.6.11` and `mmcv-full==1.6.2`:

  ```bash
  pip install -U openmim
  pip install timm==0.6.11
  mim install mmcv-full==1.6.2
  ```

- Install other requirements:

  ```bash
  pip install -r requirements.txt
  ```

- Install `clip_benchmark` using development mode:

  ```bash
  python setup.py develop
  ```

## 📦 Data Preparation

- `ImageNet-1K`: We use the standard ImageNet dataset, you can download it from [http://image-net.org/](http://image-net.org/).

- `ImageNet-A`: Download it from [https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar).

- `ImageNet-R`: Download it from [https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar).

- `ImageNetV2`: Download it from [https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz](https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz).

- `ImageNet-Sketch`: Download it using `gdown`.

  ```shell
  # GDown is needed to download the dataset. Please install it via `pip install gdown`
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

## 🔍 Linear Probing on ImageNet-1K

To train a linear classifier for `InternViT-6b` on ImageNet with 8 GPUs, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --cfg configs/intern_vit_6b_1k_224.yaml
# or manage jobs with slurm
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/intern_vit_6b_1k_224.yaml --launcher slurm
```

## 📊 Evaluation

| model name                                                     | IN-1K | IN-ReaL | IN-V2 | IN-A | IN-R | IN-Sketch |                                                                       download                                                                       |
| -------------------------------------------------------------- | :---: | :-----: | :---: | :--: | :--: | :-------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: |
| [intern_vit_6b_1k_224.yaml](configs/intern_vit_6b_1k_224.yaml) | 88.2  |  90.4   | 80.0  | 77.5 | 89.8 |   69.1    | [ckpt](https://huggingface.co/OpenGVLab/InternVL/resolve/main/intern_vit_6b_224px_head.pth) \| [log](./work_dirs/intern_vit_6b_1k_224/log_rank0.txt) |

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

# CLIP Benchmark

[![pypi](https://img.shields.io/pypi/v/clip_benchmark.svg)](https://pypi.python.org/pypi/clip_benchmark)

The goal of this repo is to evaluate CLIP-like models on a standard set
of datasets on different tasks such as zero-shot classification and zero-shot
retrieval.

Below we show the average rank (1 is the best, lower is better) of different CLIP models, evaluated
on different datasets.

![benchmark.png](benchmark.png)

The current detailed results of the benchmark can be seen [here](benchmark/README.md)
or directly in the [notebook](benchmark/results.ipynb).

## Features

- Support for zero-shot classification and zero-shot retrieval
- Support for [OpenCLIP](https://github.com/mlfoundations/open_clip) pre-trained models
- Support various datasets from [torchvision](https://pytorch.org/vision/stable/datasets.html), [tensorflow datasets](https://www.tensorflow.org/datasets), and [VTAB](https://github.com/google-research/task_adaptation).
- Support [Japanese CLIP by rinna](https://github.com/rinnakk/japanese-clip)

## How to install?

`pip install clip-benchmark`

## How to use?

To evaluate we recommend to create a models.txt like

```
ViT-B-32,openai
```

to get the list of datasets

```
wget https://raw.githubusercontent.com/LAION-AI/CLIP_benchmark/main/benchmark/webdatasets.txt
```

Then to run

```
clip_benchmark eval --pretrained_model models.txt \
    --dataset "webdatasets.txt" \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

Then to get the full table

```
clip_benchmark build benchmark_*.json --output benchmark.csv
```

### Command line interface (CLI)

The easiest way to benchmark the models is using the CLI, `clip_benchmark`.
You can specify the model to use, the dataset and the task to evaluate on. Once it is done, evaluation is performed and
the results are written into a JSON file.

### Using other models than openclip

It is possible to use other models than openclip ones. For example japanese-clip is supported

Here is an example of use

```
>>> python3 clip_benchmark/cli.py eval \
  --model_type "ja_clip" \ # flag to use japanese-clip
  --pretrained "rinna/japanese-cloob-vit-b-16" \ # now, we have `rinna/japanese-cloob-vit-b-16` or `rinna/japanese-clip-vit-b-16`.
  --language "jp" \
  --task "zeroshot_classification"  \
  --dataset "imagenet1k"  \
  --dataset_root {ROOT_PATH}

>>> cat result.json
{"dataset": "imagenet1k", "model": "ViT-B-32-quickgelu", "pretrained": "rinna/japanese-cloob-vit-b-16", "task": "zeroshot_classification", "metrics": {"acc1": 0.54636, "acc5": 0.72856, "mean_per_class_recall": 0.54522}, "language": "jp"}
```

### How to add other CLIP models

Please follow these steps:

1. Add a identity file to load model in `clip_benchmark/models`
2. Define a loading function, that returns a tuple (model, transform, tokenizer). Please see `clip_benchmark/models/open_clip.py` as an example.
3. Add the function into `TYPE2FUNC` in `clip_benchmark/models/__init__.py`

Remarks:

- The new tokenizer/model must enable to do the following things as https://github.com/openai/CLIP#usage
  - `tokenizer(texts).to(device)`  ... `texts` is a list of string
  - `model.encode_text(tokenized_texts)` ... `tokenized_texts` is a output from `tokenizer(texts).to(device)`
  - `model.encode_image(images)` ... `images` is a image tensor by the `transform`

### CIFAR-10 example

Here is an example for CIFAR-10 zero-shot classification using OpenCLIP's pre-trained model on LAION-400m:

`clip_benchmark eval --dataset=cifar10 --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-32-quickgelu --output=result.json --batch_size=64`

By default, the dataset is downloaded into `--dataset_root`, which by default is `root`.

Here is the content of `result.json` after the evaluation is done:

```json
{
    "dataset": "cifar10", "model": "ViT-B-32-quickgelu",
    "pretrained": "laion400m_e32", "task": "zeroshot_classification",
    "metrics": {"acc1": 0.9074, "acc5": 0.998}
}
```

### VOC2007 example

Here is another example with VOC2007, which is a multi-label classification dataset.

`clip_benchmark eval --dataset=voc2007_multilabel --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-32-quickgelu --output=result.json --batch_size=64`

Here is the content of `result.json` after the evaluation is done:

```json
{"dataset": "voc2007_multilabel", "model": "ViT-B-32-quickgelu", "pretrained": "laion400m_e32", "task": "zeroshot_classification", "metrics": {"mean_average_precision": 0.7627869844436646}}
```

Here, we compute the mean average precision or mAP, more details about that metric [here](https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be) in the context of multi-label classification.

### VTAB example

Here is an example on how to run it on [VTAB](https://github.com/google-research/task_adaptation) classification tasks.
First, you need to install VTAB's dedicated package.

`pip install task_adaptation==0.1`

Then, you can run it by providing the full dataset name.
Example with `eurosat`:

`clip_benchmark eval --dataset=vtab/eurosat --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-32-quickgelu --output=result.json --batch_size=64`

See [clip_benchmark/datasets/builder.py#L634](clip_benchmark/datasets/builder.py#L634) for the full list of
VTAB dataset collection.

### TensorFlow dataset example

Here is an example on how to run it on [Tensorflow datasets](https://www.tensorflow.org/datasets).
First, you need to install `tfds-nightly` and `timm`.

`pip install timm tfds-nightly`

The name of the dataset follows the template `tfds/<DATASET_NAME>`.

Example with `cifar10`:

`clip_benchmark eval --dataset=tfds/cifar10 --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-32-quickgelu --output=result.json --batch_size=64`

### COCO captions example

Here is an example for COCO captions zero-shot retrieval:

`clip_benchmark eval --dataset=mscoco_captions --task=zeroshot_retrieval --pretrained=laion400m_e32 --model=ViT-B-32-quickgelu --output=result.json --batch_size=64`

Note that for using COCO, you also need to install `pycocotools` (e.g., using `pip install pycocotools`).

### Webdataset example

Here is an example on how to run it on [webdatasets](https://github.com/webdataset/webdataset).
First, you need to install `webdataset`.

`pip install webdataset`

#### Creating a webdataset

You can either convert an already supported CLIP_benchmark dataset to webdataset format, or manually create your own with the same file structure. For already supported datasets use the CLI command `clip_benchmark_export_wds` as in this example:

```
$ clip_benchmark_export_wds --dataset cifar10 --split train --dataset_root DATA_DIR/ --output wds_cifar10/
$ clip_benchmark_export_wds --dataset cifar10 --split test --dataset_root DATA_DIR/ --output wds_cifar10/
```

which will convert the train and test splits for CIFAR-10 (downloaded to `DATA_DIR/`) and save the webdataset to `wds_cifar10/` (upload to Huggingface Hub must be done manually for now). Retrieval datasets are also supported with the `--retrieval` flag.

For other datasets, data must be stored with the following file structure:

```
root_dir/
    train/
        nshards.txt
        0.tar
        1.tar
        ...
    test/
        nshards.txt
        0.tar
        ...
    classnames.txt
    zeroshot_classification_templates.txt
    dataset_type.txt
```

Each split should be contained in its own folder and `nshards.txt` should contain a single integer corresponding to the number of TAR files. The TAR files should follow webdataset format, with an image file (.webp, .png, or .jpg) and a label (.cls) for each example. Classnames and templates are required for zeroshot classification evaluation, with each classname or template on its own line. Dataset type is required for distinguishing zeroshot retrieval evaluation: the file should just contain the text `retrieval`.

#### Evaluating on a webdataset

The name of the dataset follows the template `wds/<DATASET_NAME>`. Note that the dataset name currently only affects the name in the results output - classnames and templates are loaded directly from the included files. The dataset root directory can be either a local path to the `root_dir` as specified above, or an HTTP URL pointing to a Huggingface Hub dataset file tree.

Example with `vtab/cifar10`:

```
$ clip_benchmark eval --dataset wds/vtab/cifar10 --dataset_root ROOT_DIR/wds_vtab-cifar10/
$ clip_benchmark eval --dataset wds/vtab/cifar10 --dataset_root https://huggingface.co/datasets/clip-benchmark/wds_vtab-cifar10/tree/main
```

All other arguments remain the same as in the other examples. See `https://huggingface.co/clip-benchmark` for a full list of datasets that have already been uploaded to Huggingface.

## Evaluate mulitple models on multiple datasets

For the purpose of benchmarking, it is possible to run the CLI with multiple
pre-trained models on multiple datasets.

### Pretrained models and datasets list as arguments

For models, we can provide list of pretrained model names in the form of 'model,pretrained' (so `model` and `pretrained` are comma separated). For datasets, we can provide a list of datasets.  For languages, we can provide a list of languages.
Example:

```bash
clip_benchmark eval --pretrained_model  ViT-B-32-quickgelu,laion400m_e32 ViT-L-14,laion400m_e32  \
--dataset cifar10 cifar100 --dataset_root "clip_benchmark_datasets/{dataset}" --language en jp \
 --output "{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

Note that `--dataset_root` and `--output` can be now in the form of a template that depends on the dataset/model/language/task (for `--output`) and dataset name (for `--dataset_root`).

Note that If the benchmark fails at some point, it is possible to resume it by skipping already evaluated models using `--skip_existing`.

### Pretrained models and datasets list as files

We can also provide a path to files with models (each line is in the form of 'model,pretrained' where `model` and `pretrained` are comma separated) and datasets list (one dataset per line):

```bash
clip_benchmark eval --pretrained_model  benchmark/models.txt \
--dataset benchmark/datasets.txt --dataset_root "clip_benchmark_datasets/{dataset}"  \
 --output "{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

Examples are available in [benchmark/datasets.txt](benchmark/datasets.txt) and [benchmark/models.txt](benchmark/models.txt)

### Model and dataset collections

We can also provide model collection names (`openai`, `openclip_base`, `openclip_multilingual`, `openclip_full` are supported) or dataset collection names (`vtab`, `vtab+`, `retrieval`, `imagenet_robustness` are supported):

```bash
clip_benchmark eval --pretrained_model openai openclip_base  --dataset vtab+ retrieval \
--dataset_root "clip_benchmark_datasets/{dataset}" --not quiet \
--output "{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

See [clip_benchmark/models.py#L6](clip_benchmark/models.py#L6) and [clip_benchmark/datasets/builder.py#L634](clip_benchmark/datasets/builder.py#L634) for more information
about the collections.

### Development

For development, you can also do this:

```bash
git clone https://github.com/LAION-AI/CLIP_benchmark
cd CLIP_benchmark
python setup.py install
```

### EVA-02 examples

```shell
python3 clip_benchmark/cli.py eval --model_type open_clip --pretrained=laion2b_s9b_b144k --model=EVA02-E-14-plus --language "en" --task "zeroshot_classification" --dataset "imagenet-a" --dataset_root ./imagenet/imagenet-a/
```

### InternVL examples

- ImageNet

  ```shell
  # English
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  # Chinese
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "cn" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  # Italian
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "it" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  # Japanese
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "jp" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  # Arabic
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "ar" --task "zeroshot_classification" --dataset "imagenet1k" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  ```

- ImageNet-V2

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "imagenetv2" --dataset_root ./imagenet/imagenetv2/ --model exp33 --pretrained 399
  ```

- ImageNet-Sketch

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "imagenet_sketch" --dataset_root ./imagenet/sketch/ --model exp33 --pretrained 399
  ```

- ImageNet-A

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "imagenet-a" --dataset_root ./imagenet/imagenet-a/ --model exp33 --pretrained 399
  ```

- ImageNet-R

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "imagenet-r" --dataset_root ./imagenet/imagenet-r/ --model exp33 --pretrained 399
  ```

- ObjectNet

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "objectnet" --dataset_root ./imagenet/objectnet-1.0/ --model exp33 --pretrained 399
  ```

- CIFAR-10

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "cifar10" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  ```

- CIFAR-100

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "cifar100" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  ```

- MNIST

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "mnist" --dataset_root ./imagenet/mnist --model exp33 --pretrained 399
  ```

- SUN397

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "sun397" --dataset_root ./imagenet/sun397 --model exp33 --pretrained 399
  ```

- Food101

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "food101" --dataset_root ./imagenet/food101 --model exp33 --pretrained 399
  ```

- GTSRB

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "gtsrb" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  ```

- Flowers

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "vtab/flowers" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  ```

- FER2013

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "fer2013" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  ```

- Rendered SST2

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_classification" --dataset "renderedsst2" --dataset_root ./imagenet/ --model exp33 --pretrained 399
  ```

- Flickr30K Retrieval

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_retrieval" --dataset "flickr30k" --dataset_root ./imagenet/flickr30k_new --output flickr_retrieval.json --model exp33 --pretrained 399
  ```

- COCO Retrieval

  ```shell
  python3 clip_benchmark/cli.py eval --model_type internvl_clip --language "en" --task "zeroshot_retrieval" --dataset "mscoco_captions" --dataset_root ./imagenet/mscoco_captions --output coco_retrieval.json --model exp33 --pretrained 399
  ```

## Credits

- Thanks to [OpenCLIP](https://github.com/mlfoundations/open_clip) authors, zero-shot accuracy code is adapted from there and pre-trained models are used in the command line interface.
- Thanks to [SLIP](https://github.com/facebookresearch/SLIP) authors, some zero-shot templates and classnames are from there.
- Thanks to [Wise-ft](https://github.com/mlfoundations/wise-ft) authors, Imagenet robustness datasets code is adapted from there
- Thanks to [LiT](https://arxiv.org/abs/2111.07991.pdf) authors, some zero-shot templates and classnames of VTAB datasets are from there.
- This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template. Thanks to the author.
