# How to fine-tune the Mini-InternVL-Chat series on a custom dataset?

## 1. Prepare the pre-trained model

Before you start the second fine-tuning, you need to download the models we released. Here we provide two models:
[Mini-InternVL-Chat-2B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5) and [Mini-InternVL-Chat-4B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5).
You can use the following command to download one of them.

```bash
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/Mini-InternVL-Chat-2B-V1-5 --local-dir path/to/Mini-InternVL-Chat-2B-V1-5
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/Mini-InternVL-Chat-4B-V1-5 --local-dir path/to/Mini-InternVL-Chat-4B-V1-5
```

## 2. Prepare datasets

### Prepare released training Datasets

See [\[link\]](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets)

### Prepare your customized data

The annotation of your customized data should be filled in as a JSONL file in the following format.

```json
{"id": 0, "image": "image path relative to dataset path", "conversations": [{"from": "human", "value": "<image>\nyour question"}, {"from": "gpt", "value": "response"}]}
```

If you want to train with your customized SFT data,  it is recommended to merge your data with our internvl_1_2_finetune data by adding your data's metadata to our JSON file.
The format for organizing this JSON file is:

```
 {
     ...,
     "your_new_dataset_1": {
          "root": "path/to/images",
          "annotation": "path/to/annotation_file",
          "data_augment": true, # wether use data augment
          "repeat_time": 1,     # repeat time
          "length": 499712      # total numbers of data samples
    },
     "your_new_dataset_2": {
          "root": "path/to/images",
          "annotation": "path/to/annotation_file",
          "data_augment": true, # wether use data augment
          "repeat_time": 1,     # repeat time
          "length": 20000       # total numbers of data samples
    }
  }
```

## 3. Start finetuning

You can fine-tune our released models using this [script (Mini-InternVL-Chat-2B-V1-5)](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/internlm2_1_8b_dynamic/internvl_chat_v1_5_internlm2_1_8b_dynamic_res_finetune.sh) or [script(Mini-InternVL-Chat-4B-V1-5)](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/shell/phi3_3_8b_dynamic/internvl_chat_v1_5_phi3_3_8b_dynamic_res_finetune.sh).
Before fine-tuning, you should set the `--meta_path` to the path of the JSON file you created in the last step. And, You should change `--model_name_or_path` in these shell scripts to `path/to/Mini-InternVL-Chat-2B-V1-5`  or `path/to/Mini-InternVL-Chat-4B-V1-5`.

```bash
# using 16 GPUs with slurm system, fine-tune the full LLM
cd ./InternVL/internvl_chat/
# Mini-InternVL-Chat-2B-V1-5 
PARTITION='your partition' GPUS=16 sh shell/internlm2_1_8b_dynamic/internvl_chat_v1_5_internlm2_1_8b_dynamic_res_finetune.sh
# Mini-InternVL-Chat-4B-V1-5 
PARTITION='your partition' GPUS=16 sh shell/phi3_3_8b_dynamic/internvl_chat_v1_5_phi3_3_8b_dynamic_res_finetune.sh
```

If you see the following log in the terminal, it means the training is started successfully.

![84a0498a-b565-4964-be0f-f91d2cec7612](https://github.com/G-z-w/InternVL/assets/95175307/d66a2c40-be4c-42c8-babf-052621d2995e)

## 4. Evaluate
See [\[link\]](https://github.com/OpenGVLab/InternVL/blob/main/document/how_to_evaluate_internvl_chat_v1_5.md)
