# Blog

## InternVL-Chat-V1.2

> Date: 2024/02/12<br>
> Developed by: Zhe Chen, Weiyun Wang, Wenhai Wang

In January 2024, we released [InternVL-Chat-V1.1](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-1), featuring a structure similar to LLaVA, including a ViT, an MLP projector, and an LLM. In that version, we explored increasing the resolution to 448x448, enhancing OCR capabilities, and improving support for Chinese conversations. However, it still lagged behind existing SOTA in some benchmarks.

<img width="600" alt="image" src="https://github.com/czczup/InternVL-MoE/assets/23737120/9b68aa35-40fd-4e81-9595-d404cbbfc6bd">

Today, we are excited to introduce InternVL-Chat-V1.2. Inspired by [LLaVA-NeXT-34B](https://llava-vl.github.io/blog/2024-01-30-llava-next/), we have also adopted [Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) as the language model.
From the experimental results, **we've observed that a stronger language model (34B) can better leverage the powerful capabilities of our vision foundation model ([InternViT-6B](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2)).**

For better training reproducibility, we follow the minimalist design and data efficiency similar to LLaVA-NeXT. To reduce training costs, we provide a pre-trained MLP projector and only employ around 1 million visual instruction tuning samples for SFT. Our model has a total of 40 billion parameters and can be trained within 1.5 days using 32 A100 GPUs. The code, data, and model will be made publicly available.

### Data Preparation

Inspired by LLaVA-NeXT, we adopted a data-efficient SFT strategy to train InternVL-Chat-V1.2, utilizing approximately 1.2M of visual instruction tuning samples in total, all of which are fully open-source. In a macro sense, we build upon [ShareGPT-4V](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md#prepare-images) and additionally integrate [LLaVA-ZH](https://huggingface.co/datasets/openbmb/llava_zh), [DVQA](https://github.com/kushalkafle/DVQA_dataset), [ChartQA](https://github.com/vis-nlp/ChartQA), [AI2D](https://allenai.org/data/diagrams), [DocVQA](https://www.docvqa.org/datasets), [GeoQA+](https://github.com/SCNU203/GeoQA-Plus), and [SynthDoG-EN](https://huggingface.co/datasets/naver-clova-ix/synthdog-en). Most of the data remains consistent with LLaVA-NeXT.

For more details about data preparation, please see [here](./internvl_chat#prepare-training-datasets).

### Performance

\* Proprietary Model

| name               | image size | MMMU<sub>val</sub> | MMMU<sub>test</sub> | MathVista | MMVP | MMB  | MMB-CN | ScienceQA | POPE | TextVQA | SEED-IMG | ChartQA | AI2D | VizWiz | GQA  | MM-Vet | MME      |
| ------------------ | ---------- | ------------------ | ------------------- | --------- | ---- | ---- | ------ | --------- | ---- | ------- | -------- | ------- | ---- | ------ | ---- | ------ | -------- |
| GPT-4V\*           | unknown    | 56.8               | 55.7                | 49.9      | 38.7 | 77.0 | 73.9   | -         | -    | 78.0    | 71.6     | 78.5    | 78.2 | -      | -    | 67.7   | 1409/517 |
| Gemini Ultra\*     | unknown    | 59.4               | -                   | 53.0      | -    | -    | -      | -         | -    | 82.3    | -        | 80.8    | 79.5 | -      | -    | -      | -        |
| Gemini Pro\*       | unknown    | 47.9               | -                   | 45.2      | 40.7 | 73.6 | 74.3   | -         | -    | 74.6    | 70.7     | 74.1    | 73.9 | -      | -    | 64.3   | 1497/437 |
| Qwen-VL-Plus\*     | unknown    | 45.2               | 40.8                | 43.3      | -    | 67.0 | 70.7   | -         | -    | 78.9    | 65.7     | 78.1    | 75.9 | -      | -    | -      | 1681/502 |
| Qwen-VL-Max\*     | unknown    |  51.4              |  -               | 51.0      | -    | - | 75.1   | -         | -    | 79.5    | -     | 79.8    | 79.3  | -      | -    | -      | - |
|                    |            |          |                    |                     |           |      |      |        |           |      |         |          |         |      |        |      |        |
| LLaVA-NEXT-34B     | 672x672    | 51.1               | 44.7                | 46.5      | -    | 79.3 | 79.0   | 81.8      | 87.7 | 69.5    | 75.9     | -       | -    | 63.8   | 67.1 | 57.4   | 1631/397 |
| InternVL-Chat-V1.2 | 448x448    | 51.6               | 46.2                | 47.7      | 56.7 | 82.2 |  81.2  | 83.3      | 88.0 | 69.7    | TODO     | 67.8    | 71.6 | 60.0   | 64.0 | 48.9   | 1672/509 |

- In most benchmarks, InternVL-Chat-V1.2 achieves better performance than LLaVA-NeXT-34B.
  
### Training (SFT)

We provide [slurm scripts](./internvl_chat/shell/hermes2_yi34b/internvl_chat_v1_2_hermes2_yi34b_448_finetune.sh) for multi-node multi-GPU training. You can use either 32 or 64 GPUs to train this model. If you use 64 GPUs, training will take approximately 18 hours.

For more details about training, please see [here](./internvl_chat#start-training).

The hyperparameters used for finetuning are listed in the following table.

| Hyperparameter     | Trainable Param  | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| ------------------ | ---------------- | ----------------- | ------------- | ------ | ---------- | ------------ |
| InternVL-Chat-V1.2 | 40B (full model) | 512               | 1e-5          | 1      | 2048       | 0.05         |
