# Enhancing the Reasoning Ability of Multimodal Large Language Models via Preference Optimization

## Introduction

Existing open-source multimodal large language models (MLLMs) typically follow a two-stage training process: pre-training and supervised fine-tuning. However, these models suffer from distribution shifts, limiting their multimodal reasoning, particularly in Chain of Thought (CoT) performance.

To address this, we introduce a preference optimization (PO) process to enhance the multimodal CoT capabilities of MLLMs.
Specifically, (1) on the model side, we explore integrating PO with current MLLMs, developing a simple yet effective method that boosts multimodal CoT performance without requiring a reward model. 
(2) On the data side, we design an automated preference data construction pipeline to create [MM-Reasoning](https://huggingface.co/datasets/OpenGVLab/MM-Reasoning), a high-quality and large-scale multimodal reasoning preference dataset.

Our approach demonstrates effectiveness across multiple benchmarks, especially in multimodal reasoning tasks.
Notably, our model, [InternVL2-8B-RPO](https://huggingface.co/OpenGVLab/InternVL2-8B), achieves an accuracy of 66.2 on MathVista, outperforming InternVL2-8B by 7.9 points and even surpassing the 10$\times$ larger InternVL2-76B (65.5 points).

## MM-Reasoning Dataset

MM-Reasoning is a large-scale and high-quality multimodal reasoning preference dataset. This dataset includes both open-ended general data and Chain-of-Thought (CoT) reasoning data.

The data construction pipeline is open-sourced, see more details in our paper and [document](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html#generate-additional-preference-data).

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/619507e7b74b6c591f794340/vzRa4RfzuerxrFNdj8sIp.jpeg)
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/619507e7b74b6c591f794340/PMt4bVowh01nL9MlJk_pp.jpeg)


## Models and Performance

We finetune InternVL2-8B with RPO using this dataset.
The resulting model, [InternVL2-8B-RPO](https://huggingface.co/OpenGVLab/InternVL2-8B-RPO), exhibits enhanced multimodal reasoning abilities and reduced hallucinations compared to InternVL2-8B.

| Model Name              | M3CoT | MathVista | MathVision MINI | MMVet (GPT4-Turbo) | LLaVA-Bench | POPE  | CRPE  | MMHalBench |
| ----------------------- | :---: | :-------: | :-------------: | :----------------: | :---------: | :---: | :---: | :--------: |
| Gemini-1.5-Pro          |   -   |   63.9    |      19.2       |         -          |      -      |   -   |   -   |     -      |
| GPT-4o                  | 64.3  |   63.8    |      30.4       |        69.1        |    97.6     | 86.9  | 76.6  |    4.0     |
| GPT-4o-Mini             | 61.9  |   52.4    |      27.3       |        66.9        |    95.4     | 85.1  | 73.1  |    3.6     |
| LLaVA-1.5-13B           | 39.5  |   27.6    |      11.1       |        36.3        |    70.7     | 85.9  | 55.6  |    2.4     |
| Qwen2-VL-7B             | 57.8  |   58.2    |      21.1       |        60.6        |    67.7     | 88.1  | 74.4  |    3.4     |
| MiniCPM-V-2-6-8B        | 56.0  |   60.6    |      23.4       |        57.4        |    83.4     | 87.3  | 75.2  |    3.6     |
| LLaVA-OneVision-7B      | 52.3  |   63.2    |      18.4       |        51.4        |    79.9     | 88.4  | 73.7  |    3.1     |
| InternVL2-26B           | 58.2  |   59.4    |      23.4       |        62.1        |    92.3     | 88.0  | 75.6  |    3.7     |
| InternVL2-40B           | 63.6  |   63.7    |      21.4       |        65.5        |    100.5    | 88.4  | 77.3  |    3.9     |
| InternVL2-76B           | 65.4  |   65.5    |      23.7       |        65.7        |    99.3     | 89.0  | 77.8  |    3.8     |
| InternVL2-Pro           | 65.6  |   66.3    |      18.8       |        69.4        |    99.5     | 88.2  | 77.6  |    3.7     |
| InternVL2-8B            | 59.3  |   58.3    |      20.4       |        54.2        |    73.2     | 86.9  | 75.5  |    3.3     |
| InternVL2-8B-RPO (ours) | 79.2  |   66.2    |      25.7       |        56.2        |    76.7     | 88.1  | 75.4  |    3.5     |

We evaluate the models using the InternVL and VLMEvalKit repositories. Specifically, M3CoT, MathVista, POPE, and MMHalBench are tested with InternVL, while MathVision, LLaVA-Bench, MMVet, and CRPE are evaluated using VLMEvalKit.

## Train

Please refer to [our document](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html) for more details about how to train with our data.

## Citation
If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```
