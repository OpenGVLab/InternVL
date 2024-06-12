# How to fine-tune InternVL-Chat-V1.2 on a custom dataset?

#### 1. Prepare the Pre-trained Model

Before you start the second fine-tuning, you need to download the pre-trained model we provided. Here we provide two pre-trained models, [InternVL-Chat-V1.2](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2) and [InternVL-Chat-V1.2-Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus).

You can use the following command to download one of them, we recommend you download the plus version.

```shell
cd pretrained/
# pip install -U huggingface_hub
# download OpenGVLab/InternVL-Chat-V1-2
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-2 --local-dir InternVL-Chat-V1-2
# download OpenGVLab/InternVL-Chat-V1-2-Plus
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-2-Plus --local-dir InternVL-Chat-V1-2-Plus
```

#### 2. Prepare Your Custom Training Data

After downloading the pre-trained model, you need to prepare your customized SFT data. You should write a JSON file in `internvl_chat/shell/data/`, just like [this file](./shell/data/internvl_1_2_finetune.json).

The format for organizing this JSON file is:

```json
{
  "your-custom-dataset-1": {
    "root": "path/to/the/image/",
    "annotation": "path/to/the/jsonl/annotation",
    "data_augment": false,
    "repeat_time": 1,
    "length": number of your data
  },
  ...
}
```

For example:

```json
{
  "sharegpt4v_instruct_gpt4-vision_cap100k": {
    "root": "playground/data/",
    "annotation": "playground/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 102025
  }
}
```

#### 3. Start Fine-tuning

You can fine-tune our pre-trained models using this [script (train full LLM)](../internvl_chat/shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune_continue.sh) or this [script (train LoRA adapter)](../internvl_chat/shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune_continue_lora.sh), depending on your available GPU devices.

Before fine-tuning, you should set the `--meta_path` to the path of the JSON file you created in the last step. And, the default pre-trained model in these shell scripts is `./pretrained/InternVL-Chat-V1-2`. You should change it to `./pretrained/InternVL-Chat-V1-2-Plus` if you want to fine-tune our plus version.

> Note: fine-tune the full LLM needs 16 A100 80G GPUs, and fine-tune the LoRA needs 2 A100 80G GPUs.

```sh
# using 16 GPUs with slurm system, fine-tune the full LLM
PARTITION='your partition' GPUS=16 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune_continue.sh
# using 2 GPUs, fine-tune the LoRA
CUDA_VISIBLE_DEVICES=0,1 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune_continue_lora.sh
```

If you run into any problems, please let me know and I will improve the training guide to make it easier to use.
