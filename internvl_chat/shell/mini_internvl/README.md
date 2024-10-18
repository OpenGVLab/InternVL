# Mini-InternVL

## Abstract

We introduce Mini-InternVL, a series of MLLMs with parameters ranging from 1B to 4B, which achieves 90\% of the performance with only 5\% of the parameters. This significant improvement in efficiency and effectiveness makes our models more accessible and applicable in various real-world scenarios. To further promote the adoption of our models, we develop a unified adaptation framework for Mini-InternVL, which enables our models to transfer and outperform specialized models in downstream tasks, including autonomous driving, medical imaging, and remote sensing. We believe that our models can provide valuable insights and resources to advance the development of efficient and effective MLLMs.

放图

## Models and Performance

| Model   |  MMMU (val)| MathVista (testmini) |AI2D |ChartQA |DocVQA | InfoVQA |OCRBench| MMB-EN | MMB-CN |Avg. Score |
|:--------:|:-----:|:-----:|:-----: |:------:|:------:|:----:|:----:|:------:|----:| :------:|
|Claude3.5-Sonnet|65.9 | 67.7| 94.7 | 90.8 | 95.2 | -  | 788 | 79.7 | 80.7 | 81.7 |
|InternVL2-Llama3-76B|58.2  | 65.5  | 87.6 | 88.4 | 94.1 | 82.0  | 839  | 86.5 | 86.3  | 81.4 |
|Mini-InternVL-1B ([🤗](https://huggingface.co/OpenGVLab/InternVL2-1B))| 36.7 | 37.7 | 64.1 | 72.9  | 81.7  | 50.9 | 754  | 65.4 | 60.7  | 60.6 (74\%) |
|Mini-InternVL-2B ([🤗](https://huggingface.co/OpenGVLab/InternVL2-2B))|36.3| 46.3  | 74.1 | 76.2 | 86.9 | 58.9 | 784 | 73.2 | 70.9 | 66.8 (82\%)|
|Mini-InternVL-4B ([🤗](https://huggingface.co/OpenGVLab/InternVL2-4B))|48.3 | 58.6  | 78.9 | 81.5  | 89.2 | 67.0 | 788  | 78.6 | 73.9 | 72.8 (90\%) |

- We evaluate models using InternVL and VLMEvalKit repositories. AI2D, ChartQA, DocVQA, InfoVQA, and MMBench are tested with InternVL, while MathVistaand OCRBench use VLMEvalKit. For MMMU, we report scores from OpenCompass leaderboard.

- The Avg. Score is the average of the scores from all tested benchmarks, with the OCR-Bench score divided by 10. The values in parentheses represent the relative parameters and performance of Mini-InternVL compared to *InternVL2-Llama3-76B*, which is considered as 100\%.

## Domain Adaptation

放图

In the [document](http://xxx.xxx.xxx), we provide detailed information on the datasets and the fine-tuning process.

## Citation

## Acknowledgements

[DriveGPT4](https://tonyxuqaq.github.io/projects/DriveGPT4/),
[GeoChat](https://github.com/mbzuai-oryx/GeoChat),
[SkySenseGPT](https://github.com/Luo-Z13/SkySenseGPT),
[DriveLM](https://github.com/OpenDriveLab/DriveLM)

