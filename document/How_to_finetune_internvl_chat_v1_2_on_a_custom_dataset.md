# How to Fine-tune InternVL-Chat-V1-2 on a Custom Dataset

## 1. Prepare the Pre-trained Model

Before starting the second fine-tuning process, download the pre-trained model we provide. Two versions are available: [InternVL-Chat-V1-2](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2) and [InternVL-Chat-V1-2-Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus). We recommend downloading the Plus version.

Use the following commands to download the desired model:

```shell
cd pretrained/
# pip install -U huggingface_hub
# Download OpenGVLab/InternVL-Chat-V1-2
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-2 --local-dir InternVL-Chat-V1-2
# Download OpenGVLab/InternVL-Chat-V1-2-Plus
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL-Chat-V1-2-Plus --local-dir InternVL-Chat-V1-2-Plus
```

## 2. Prepare Your Customized Training Data

After downloading the pre-trained model, prepare your customized SFT (Supervised Fine-Tuning) data. Create a JSON file in `internvl_chat/shell/data/` similar to [this example](./shell/data/internvl_1_2_finetune.json).

The format for the JSON file should be:

```json
{
  "your-custom-dataset-1": {
    "root": "path/to/the/image/",
    "annotation": "path/to/the/jsonl/annotation",
    "data_augment": false,
    "repeat_time": 1,
    "length": "number of your data"
  },
  ...
}
```

Example:

```json
{
  "sharegpt4v_instruct_gpt4-vision_cap100k": {
    "root": "playground/data/",
    "annotation": "playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 102025
  }
}
```

## 3. Start Fine-tuning

Fine-tune the pre-trained models using either the [script for training the full LLM](../internvl_chat/shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune_continue.sh) or the [script for training the LoRA adapter](../internvl_chat/shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune_continue_lora.sh), depending on your available GPU resources.

Before fine-tuning, set the `--meta_path` to the path of the JSON file you created in the previous step. The default pre-trained model path in these shell scripts is `./pretrained/InternVL-Chat-V1-2`. Update it to `./pretrained/InternVL-Chat-V1-2-Plus` if you are using the Plus version.

> Note: Fine-tuning the full LLM requires 16 A100 80G GPUs, whereas fine-tuning the LoRA requires 2 A100 80G GPUs.

Commands for fine-tuning:

```sh
# Using 16 GPUs with SLURM system, fine-tune the full LLM
PARTITION='your partition' GPUS=16 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune_continue.sh
# Using 2 GPUs, fine-tune the LoRA
CUDA_VISIBLE_DEVICES=0,1 sh shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_res_finetune_continue_lora.sh
```

If you encounter any issues, please let me know, and I will update the training guide to enhance its usability.
