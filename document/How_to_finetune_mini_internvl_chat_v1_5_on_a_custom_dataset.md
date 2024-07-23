# How to Fine-tune the Mini-InternVL-Chat Series on a Custom Dataset

## 1. Prepare the Pre-trained Model

Before starting the second fine-tuning process, download the models we released. We provide two models: [Mini-InternVL-Chat-2B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5) and [Mini-InternVL-Chat-4B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5).

Use the following commands to download the desired model:

```bash
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/Mini-InternVL-Chat-2B-V1-5 --local-dir path/to/Mini-InternVL-Chat-2B-V1-5
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/Mini-InternVL-Chat-4B-V1-5 --local-dir path/to/Mini-InternVL-Chat-4B-V1-5
```

## 2. Prepare Datasets

### Prepare Released Training Datasets

Refer to [this link](../internvl_chat#prepare-training-datasets) for details on preparing released training datasets.

### Prepare Your Customized Data

Create a JSONL file with annotations for your custom data in the following format:

```json
{"id": 0, "image": "image path relative to dataset path", "conversations": [{"from": "human", "value": "<image>\nyour question"}, {"from": "gpt", "value": "response"}]}
```

If you want to train with your customized SFT data, merge your data with our [internvl_1_2_finetune](../internvl_chat/shell/data/internvl_1_2_finetune.json) data by adding your data's metadata to our [JSON file](../internvl_chat/shell/data/internvl_1_2_finetune.json). The format for organizing this JSON file is:

```json
{
  "sharegpt4v_instruct_gpt4-vision_cap100k": {
    "root": "playground/data/",
    "annotation": "playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 102025
  },
  "llava_instruct_150k_zh": {
    "root": "playground/data/coco/",
    "annotation": "playground/opensource/llava_instruct_150k_zh.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 157712
  },
  "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k": {
    "root": "playground/data/",
    "annotation": "playground/opensource/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 665058
  },
  "dvqa_train_200k": {
    "root": "playground/data/dvqa/",
    "annotation": "playground/opensource/dvqa_train_200k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 200000
  },
  "chartqa_train_18k": {
    "root": "playground/data/chartqa/",
    "annotation": "playground/opensource/chartqa_train_18k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 18317
  },
  "ai2d_train_12k": {
    "root": "playground/data/ai2d/",
    "annotation": "playground/opensource/ai2d_train_12k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 12413
  },
  "docvqa_train_10k": {
    "root": "playground/data/docvqa/",
    "annotation": "playground/opensource/docvqa_train_10k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 10211
  },
  "geoqa+": {
    "root": "playground/data/geoqa+/",
    "annotation": "playground/opensource/geoqa+.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 72318
  },
  "synthdog_en": {
    "root": "playground/data/synthdog-en/",
    "annotation": "playground/opensource/synthdog_en.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 29765
  },
  "your_new_dataset": {
    "root": "path/to/images",
    "annotation": "path/to/annotation_file",
    "data_augment": false,
    "repeat_time": 1,
    "length": 499712
  }
}
```

## 3. Start Fine-tuning

Fine-tune the released models using either the [script for Mini-InternVL-Chat-2B-V1-5](./internvl_chat/shell/internlm2_1_8b_dynamic/internvl_chat_v1_5_internlm2_1_8b_dynamic_res_finetune.sh) or the [script for Mini-InternVL-Chat-4B-V1-5](./internvl_chat/shell/phi3_3_8b_dynamic/internvl_chat_v1_5_phi3_3_8b_dynamic_res_finetune.sh). Set the `--meta_path` to the path of the JSON file you created in the last step and update `--model_name_or_path` in these shell scripts to `path/to/Mini-InternVL-Chat-2B-V1-5` or `path/to/Mini-InternVL-Chat-4B-V1-5`.

```bash
# Using 16 GPUs with SLURM system, fine-tune the full LLM
cd internvl_chat/
# Mini-InternVL-Chat-2B-V1-5
PARTITION='your partition' GPUS=16 sh shell/internlm2_1_8b_dynamic/internvl_chat_v1_5_internlm2_1_8b_dynamic_res_finetune.sh
# Mini-InternVL-Chat-4B-V1-5
PARTITION='your partition' GPUS=16 sh shell/phi3_3_8b_dynamic/internvl_chat_v1_5_phi3_3_8b_dynamic_res_finetune.sh
```

If you see the following log in the terminal, it means the training has started successfully:

![Training Started Successfully](https://github.com/G-z-w/InternVL/assets/95175307/d66a2c40-be4c-42c8-babf-052621d2995e)

For a complete example training log, refer to [this link](./training_log.txt).

## 4. Evaluate

Refer to [this link](./document/How_to_evaluate_internvl_chat_v1_5.md) for evaluation details.
