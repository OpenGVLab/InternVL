# Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance

## Introduction

We introduce Mini-InternVL, a series of MLLMs with parameters ranging from 1B to 4B, which achieves 90% of the performance with only 5% of the parameters.
This significant improvement in efficiency and effectiveness makes our models more accessible and applicable in various real-world scenarios.

![internvl 1 5_wwh_33_2](https://github.com/user-attachments/assets/820ed173-4bd1-45a6-95d6-59c1be01d53f)

- InternViT-300M

We employ InternViT-300M as our visual encoder, a lightweight vision model that inherits the capabilities of a powerful vision encoder. We directly leverage InternViT-6B that has undergone generative training on diverse datasets to transfer knowledge to a lightweight vision model, CLIP-ViT-L-336px.

- Adaptation for Mini-InternVL

To further promote the adoption of our models, we develop a unified adaptation framework for Mini-InternVL, which enables our models to transfer and outperform specialized models in downstream tasks, including autonomous driving, medical images, and remote sensing. We hope to provide insights into the application of MLLMs.

## Models and Performance

|                                 Model                                  | MMMU (val) | MathVista (testmini) | AI2D | ChartQA | DocVQA | InfoVQA | OCRBench | MMB-EN | MMB-CN | Avg. Score |
| :--------------------------------------------------------------------: | :--------: | :------------------: | :--: | :-----: | :----: | :-----: | :------: | :----: | -----: | :--------: |
|                            Claude3.5-Sonnet                            |    65.9    |         67.7         | 94.7 |  90.8   |  95.2  |    -    |   788    |  79.7  |   80.7 |    81.7    |
|                          InternVL2-Llama3-76B                          |    58.2    |         65.5         | 87.6 |  88.4   |  94.1  |  82.0   |   839    |  86.5  |   86.3 |    81.4    |
| Mini-InternVL-1B ([🤗](https://huggingface.co/OpenGVLab/InternVL2-1B)) |    36.7    |         37.7         | 64.1 |  72.9   |  81.7  |  50.9   |   754    |  65.4  |   60.7 | 60.6 (74%) |
| Mini-InternVL-2B ([🤗](https://huggingface.co/OpenGVLab/InternVL2-2B)) |    36.3    |         46.3         | 74.1 |  76.2   |  86.9  |  58.9   |   784    |  73.2  |   70.9 | 66.8 (82%) |
| Mini-InternVL-4B ([🤗](https://huggingface.co/OpenGVLab/InternVL2-4B)) |    48.3    |         58.6         | 78.9 |  81.5   |  89.2  |  67.0   |   788    |  78.6  |   73.9 | 72.8 (90%) |

- We evaluate models using InternVL and VLMEvalKit repositories. AI2D, ChartQA, DocVQA, InfoVQA, and MMBench are tested with InternVL, while MathVistaand OCRBench use VLMEvalKit. For MMMU, we report scores from OpenCompass leaderboard.

- The Avg. Score is the average of the scores from all tested benchmarks, with the OCR-Bench score divided by 10. The values in parentheses represent the relative parameters and performance of Mini-InternVL compared to *InternVL2-Llama3-76B*, which is considered as 100%.

## Domain Adaptation

Visual tasks (*e.g.* Image classification, region perception, multi-view images tasks, video related tasks and visual grounding) can be  formulated into VQA format.

![framework_03_2](https://github.com/user-attachments/assets/63bffb31-cf05-4f52-a679-4700650d0c37)

In the [document](https://internvl.readthedocs.io/en/latest/internvl2.0/domain_adaptation.html), we provide detailed information on the datasets and the fine-tuning process.

### Adaptation models

We have released the adaptation models for the following four domains. The script for evaluation is in the [document](https://internvl.readthedocs.io/en/latest/internvl2.0/domain_adaptation.html#id3).

<table>
  <tr>
    <th>Model Name</th>
    <th>HF Link</th>
    <th>Note</th>
  </tr>
  <tr>
    <td>Mini-InternVL2-DA-Drivelm</td>
    <td><a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-1B-DA-Drivelm">🤗1B</a> / <a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-2B-DA-Drivelm">🤗2B</a> / <a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-4B-DA-Drivelm">🤗4B</a></td>
    <td> Adaptation for <a href="https://github.com/OpenDriveLab/DriveLM/tree/main/challenge"> CVPR 2024 Autonomous Driving Challenge </a></td>
  </tr>
  <tr>
    <td>Mini-InternVL2-DA-BDD</td>
    <td><a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-1B-DA-BDD">🤗1B</a> / <a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-2B-DA-BDD">🤗2B</a> / <a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-4B-DA-BDD">🤗4B</a></td>
    <td> Fine-tuning with data constructed by <a href="https://tonyxuqaq.github.io/projects/DriveGPT4/"> DriveGPT4 </a></td>
  </tr>
  <tr>
    <td>Mini-InternVL2-DA-RS</td>
    <td><a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-1B-DA-RS">🤗1B</a> / <a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-2B-DA-RS">🤗2B</a> / <a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-4B-DA-RS">🤗4B</a></td>
    <td> Adaptation for remote sensing domain </td>
  </tr>
  <tr>
    <td>Mini-InternVL2-DA-Medical</td>
    <td><a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-1B-DA-Medical">🤗1B</a> / <a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-2B-DA-Medical">🤗2B</a> / <a href="https://huggingface.co/OpenGVLab/Mini-InternVL2-4B-DA-Medical">🤗4B</a></td>
    <td> Fine-tuning using our <a href="https://huggingface.co/datasets/OpenGVLab/InternVL-Domain-Adaptation-Data/blob/main/train_meta/internvl_1_2_finetune_medical.json">medical data</a>.</td>
  </tr>
</table>

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
@article{gao2024mini,
  title={Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5\% Parameters and 90\% Performance},
  author={Gao, Zhangwei and Chen, Zhe and Cui, Erfei and Ren, Yiming and Wang, Weiyun and Zhu, Jinguo and Tian, Hao and Ye, Shenglong and He, Junjun and Zhu, Xizhou and others},
  journal={arXiv preprint arXiv:2410.16261},
  year={2024}
}
```

## Acknowledgements

[DriveGPT4](https://tonyxuqaq.github.io/projects/DriveGPT4/),
[GeoChat](https://github.com/mbzuai-oryx/GeoChat),
[SkySenseGPT](https://github.com/Luo-Z13/SkySenseGPT),
[DriveLM](https://github.com/OpenDriveLab/DriveLM)
