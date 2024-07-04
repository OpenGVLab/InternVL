# <img width="60" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/47669167/7037290e-f474-4d11-b90f-1d8316087bf8"> InternVL Family: Closing the Gap to Commercial Multimodal Models with Open-Source Suites —— A Pioneering Open-Source Alternative to GPT-4o

[\[📖中文版本ReadMe\]](./README_zh.md) [\[🆕 Blog\]](https://internvl.github.io/blog/)  [\[🚀 InternVL2 Blog\]](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)  [\[📜 InternVL 1.5 Report\]](https://arxiv.org/abs/2404.16821)[\(中文解读\)](https://zhuanlan.zhihu.com/p/699439759) [\[📜 InternVL 1.0 Paper\]](https://arxiv.org/abs/2312.14238) [\[🚀 Quick Start\]](#quick-start-with-huggingface)

[\[🤗 InternVL2 HF Chat Demo\]](https://huggingface.co/spaces/OpenGVLab/InternVL) [\[🗨️ Chat Demo\]](https://internvl.opengvlab.com/)  [\[🌐 API\]](./document/How_to_use_InternVL_API.md) 

<a href="https://trendshift.io/repositories/9803" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9803" alt="OpenGVLab%2FInternVL | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>&nbsp;&nbsp;
<img height="55" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/354aa3b7-0129-49b7-94ae-c4deb755051f">


## News🚀🚀🚀
- `2024/07/04`: 🚀We are pleased to release InternVL2. It achieved a 62.0% accuracy on the MMMU Benchmark, matching the performance of leading closed-source commercial models like GPT-4o. The free API of our model can be applied by filling ([English application form](https://docs.google.com/forms/d/e/1FAIpQLSfMCzhPr1OOEKau_6jwTU0EiZMSFckDo-HMlc_hUudhF_97rw/viewform?usp=sf_link))/([中文申请表](https://wj.qq.com/s2/14910502/25a4/)). Models are available at [HF link](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e).
- `2024/06/19`: 🚀We release Needle In A Multimodal Haystack ([MM-NIAH](https://github.com/OpenGVLab/MM-NIAH)), the first benchmark designed to systematically evaluate the capability of existing MLLMs to comprehend long multimodal documents. **Experimental results show that the performance of Gemini-1.5 on tasks involving image needles is no better than random guessing.**
- `2024/06/04`: InternVL 1.5 achieved state-of-the-art in the Image MLLM category on the [Video-MME](https://github.com/BradyFU/Video-MME) dataset, demonstrating strong generalization across multiple images, surpassing many specialized Video MLLMs and nearing the top open-source video model, LLaVA-Next-Video.
- `2024/05/30`: 🚀 🚀 We release [ShareGPT-4o](https://sharegpt4o.github.io/), a groundbreaking large-scale resource that we plan to open-source with 200K meticulously annotated images, 10K videos with highly descriptive captions, and 10K audio files with detailed descriptions.
- `2024/05/29`: 🚀 We release the Mini-InternVL-Chat series, which includes two models: [Mini-InternVL-Chat-2B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5) and [Mini-InternVL-Chat-4B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5). Our small models achieve impressive performance with minimal size: the 2B model delivers 80% of the performance with only 8% of the model size, and the 4B model achieves 90% of the performance with just 16% of the model size. For more details, please check our [blog](https://internvl.github.io/blog/2024-05-25-Mini-InternVL-1.5/).
- `2024/05/28`: Thanks to the [lmdeploy](https://github.com/InternLM/lmdeploy) team for providing AWQ quantization support. The 4-bit model is available at [OpenGVLab/InternVL-Chat-V1-5-AWQ](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-AWQ).
- `2024/05/13`: 🔥 InternVL can now be used as the [text encoder](https://huggingface.co/OpenGVLab/InternVL-14B-224px) for diffusion models to support multilingual generation natively in over 110 languages worldwide. See [MuLan](https://github.com/mulanai/MuLan) for more details.
- `2024/04/28`: We release the INT8 version of InternVL-Chat-V1-5, see [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-Int8).
- `2024/04/28`: We achieve the SOTA performance (75.74) on the Infographics VQA benchmark, see [here](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=3).
- `2024/04/18`: InternVL-Chat-V1-5 has been released at [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5), approaching the performance of GPT-4V and Gemini Pro on various benchmarks like MMMU, DocVQA, ChartQA, MathVista, etc.
- `2024/02/27`: InternVL is accepted by CVPR 2024! 🎉
- `2024/02/24`: InternVL-Chat models have been included in the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
- `2024/02/21`: [InternVL-Chat-V1-2-Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) achieves SOTA performance on MathVista (59.9), MMBench (83.8), and MMVP (58.7). See our [blog](https://internvl.github.io/blog/2024-02-21-InternVL-1.2/) for more details.
- `2024/02/12`: InternVL-Chat-V1-2 has been released. It achieves 51.6 on MMMU val and 82.3 on MMBench test. For more details, please refer to our [blog](https://internvl.github.io/blog/2024-02-21-InternVL-1.2/), [SFT data](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets) or try our [demo](https://internvl.opengvlab.com/). The model is now available on [HuggingFace](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2), and both training/evaluation data and scripts are open-sourced.
- `2024/02/04`: [InternVL-Chat-V1-1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1) achieves 44.67% on [MMVP](https://github.com/tsb0601/MMVP), higher than GPT-4V!
- `2024/01/27`: We release 448 resolution model, achieving 76.6 on MMBench dev, see [here](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#-evaluation-chinese-models).
- `2024/01/24`: InternVL-Chat-V1-1 is released, it supports Chinese and has stronger OCR capability, see [here](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1).
- `2024/01/16`: We release our [customized mmcv/mmsegmentation/mmdetection code](https://github.com/OpenGVLab/InternVL-MMDetSeg), integrated with DeepSpeed, which can be used for training large-scale object detection and semantic segmentation models.

## Documents

- Installation

  - How to install the environment? [\[link\]](./INSTALLATION.md)

- Training or Fine-tuning

  - How to reproduce the SFT stage of InternVL-Chat-V1-2? [\[link\]](./internvl_chat#start-training)
  - How to fine-tune InternVL-Chat-V1-2 on a custom dataset? [\[link\]](./document/How_to_finetune_internvl_chat_v1_2_on_a_custom_dataset.md)
  - How to fine-tune the Mini-InternVL-Chat series on a custom dataset? [\[link\]](./document/How_to_finetune_mini_internvl_chat_v1_5_on_a_custom_dataset.md)

- Benchmark Test

  > Due to minor implementation differences between this codebase and VLMEvalKit, slight discrepancies in performance metrics may occur when testing the same model.

  - How to evaluate InternVL-Chat-V1-5? [\[link\]](./document/How_to_evaluate_internvl_chat_v1_5.md)
  - How to evaluate InternVL-Chat-V1-5 using VLMEvalKit? (Recommend) [\[link\]](./document/How_to_evaluate_internvl_chat_v1_5_using_vlmevalkit.md)
  - How to evaluate Mini-InternVL-Chat-2B-V1-5 using VLMEvalKit? (Recommend) [\[link\]](./document/How_to_evaluate_mini_internvl_chat_2b_v1_5_using_vlmevalkit.md)
  - How to evaluate Mini-InternVL-Chat-4B-V1-5 using VLMEvalKit? (Recommend) [\[link\]](./document/How_to_evaluate_mini_internvl_chat_4b_v1_5_using_vlmevalkit.md)

- Deployment
  - How to use InternVL API? [\[link\]](./document/How_to_use_InternVL_API.md)
  - How to deploy a local demo? [\[link\]](./document/How_to_deploy_a_local_demo.md)
  - How to run InternVL-1.5 8bit with Nvidia V100 GPU? [\[link\]](https://github.com/OpenGVLab/InternVL/issues/144) [\[中文教程\]](https://zhuanlan.zhihu.com/p/697188143)
  - How to perform batch inference? [\[link\]](https://github.com/OpenGVLab/InternVL/blob/main/README.md?plain=1#L617)
  - Inference Acceleration by LMDeploy [\[link\]](#inference-acceleration-by-lmdeploy) [\[中文教程\]](https://zhuanlan.zhihu.com/p/696955211)

## Compared with SOTA VLLMs

<p align="center"><img width="1000" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/8529570/7c31c8f0-d11b-43ec-845d-70186d6c4ef3"></p>


## What is InternVL?

InternVL scales up the ViT to _**6B parameters**_ and aligns it with LLM.

## Model Zoo

**Vision Large Language Model**

| Model                      | Date       | Download                                                                             | Note                                                                                                                                                               |
| -------------------------------- | ---------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| InternVL2     | 2024.07.04 | 🤗 [HF link](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e)                | achieving performance close to GPT-4o                                                                                                                        |
| Mini&#8209;InternVL&#8209;Chat&#8209;4B&#8209;V1&#8209;5 | 2024.05.28 | 🤗 [HF link](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5)            | 🚀🚀 16% of the model size, 90% of the performance                                                                                   |
| Mini-InternVL-Chat-2B-V1-5 | 2024.05.19 | 🤗 [HF link](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5)            | 🚀 8% of the model size, 80% of the performance                                                                                                                    |
| InternVL-Chat-V1-5-AWQ     | 2024.05.28 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-AWQ)                | The 4-bit version of InternVL-Chat-V1-5                                                                                                                            |
| InternVL-Chat-V1-5-Int8    | 2024.04.28 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-Int8)               | The 8-bit version of InternVL-Chat-V1-5                                                                                                                            |
| InternVL-Chat-V1-5         | 2024.04.18 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)                    | support 4K image; super strong OCR; Approaching the performance of GPT-4V and Gemini Pro on various benchmarks like MMMU, DocVQA, ChartQA, MathVista, etc. (🔥new) |
| InternVL-Chat-V1-2-Plus    | 2024.02.21 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus)               | more SFT data and stronger                                                                                                                                         |
| InternVL-Chat-V1-2         | 2024.02.11 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2)                    | scaling up LLM to 34B                                                                                                                                              |
| InternVL-Chat-V1-1         | 2024.01.24 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1)                    | support Chinese and stronger OCR                                                                                                                                   |
| InternVL-Chat-19B-448px    | 2024.02.03 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B-448px) | 448 resolution                                                                                                                                                     |
| InternVL-Chat-19B          | 2023.12.25 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B)       | English multimodal dialogue                                                                                                                                        |
| InternVL-Chat-13B          | 2023.12.25 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B)        | English multimodal dialogue                                                                                                                                        |

**Vision-Language Foundation Model**

| Model                   | Date       | Download                                                               | Note                                                                                                   |
| ----------------------- | ---------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| InternViT-300M-448px    | 2024.05.25 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-300M-448px)    | distilled small vision foundation model with 300M parameters (🔥new)                                   |
| InternViT-6B-448px-V1-5 | 2024.04.20 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) | support dynamic resolution, super strong OCR (🔥new)                                                   |
| InternViT-6B-448px-V1-2 | 2024.02.11 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2) | 448 resolution                                                                                         |
| InternViT&#8209;6B&#8209;448px&#8209;V1&#8209;0 | 2024.01.30 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-0) | 448 resolution                                                                                         |
| InternViT-6B-224px      | 2023.12.22 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-224px)      | vision foundation model                                                                                |
| InternVL-14B-224px      | 2023.12.22 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-14B-224px)      | vision-language foundation model, InternViT-6B + QLLaMA, can be used for image-text retrival like CLIP |

**InternVL-2 API**<br>
We encourage everyone to use our API for research. For better management, please submit ([English application form](https://docs.google.com/forms/d/e/1FAIpQLSfMCzhPr1OOEKau_6jwTU0EiZMSFckDo-HMlc_hUudhF_97rw/viewform?usp=sf_link))/([中文申请表](https://wj.qq.com/s2/14910502/25a4/)) to obtain free API access.

## What can InternVL do?

<details>
  <summary>Visual Perception (click to expand)</summary>

- Linear-Probe Image Classification [\[see details\]](./classification#-evaluation)

  ViT-22B uses the private JFT-3B dataset.

  | method              | #param | IN-1K | IN-ReaL | IN-V2 | IN-A | IN-R | IN-Sketch |
  | ------------------- | :----: | :---: | :-----: | :---: | :--: | :--: | :-------: |
  | OpenCLIP-G          |  1.8B  | 86.2  |  89.4   | 77.2  | 63.8 | 87.8 |   66.4    |
  | DINOv2-g            |  1.1B  | 86.5  |  89.6   | 78.4  | 75.9 | 78.8 |   62.5    |
  | EVA-01-CLIP-g       |  1.1B  | 86.5  |  89.3   | 77.4  | 70.5 | 87.7 |   63.1    |
  | MAWS-ViT-6.5B       |  6.5B  | 87.8  |    -    |   -   |  -   |  -   |     -     |
  | ViT-22B\*           | 21.7B  | 89.5  |  90.9   | 83.2  | 83.8 | 87.4 |     &#8209;     |
  | InternViT-6B (ours) |  5.9B  | 88.2  |  90.4   | 79.9  | 77.5 | 89.8 |   69.1    |

- Semantic Segmentation [\[see details\]](./segmentation#-evaluation)

  | method                | decoder | #param (train/total) | crop size | mIoU         |
  | --------------------- | :-----: | :------------------: | :-------: | ------------ |
  | OpenCLIP-G (frozen)   | Linear  |     0.3M / 1.8B      |    512    | 39.3         |
  | ViT-22B (frozen)      | Linear  |     0.9M / 21.7B     |    504    | 34.6         |
  | InternViT-6B (frozen) | Linear  |     0.5M / 5.9B      |    504    | 47.2 (+12.6) |
  | ViT-22B (frozen)      | UperNet |     0.8B / 22.5B     |    504    | 52.7         |
  | InternViT-6B (frozen) | UperNet |     0.4B / 6.3B      |    504    | 54.9 (+2.2)  |
  | ViT-22B               | UperNet |    22.5B / 22.5B     |    504    | 55.3         |
  | InternViT-6B          | UperNet |     6.3B / 6.3B      |    504    | 58.9 (+3.6)  |

- Zero-Shot Image Classification [\[see details\]](./clip_benchmark#imagenet-variants-and-objectnet)

  | method            | IN-1K | IN-A | IN-R | IN-V2 | IN-Sketch | ObjectNet |
  | ----------------- | :---: | :--: | :--: | :---: | :-------: | :-------: |
  | OpenCLIP-G        | 80.1  | 69.3 | 92.1 | 73.6  |   68.9    |   73.0    |
  | EVA-02-CLIP-E+    | 82.0  | 82.1 | 94.5 | 75.7  |   71.6    |   79.6    |
  | ViT-22B\*         | 85.9  | 90.1 | 96.0 | 80.9  |     &#8209;     |   87.6    |
  | InternVL-C (ours) | 83.2  | 83.8 | 95.5 | 77.3  |   73.9    |   80.6    |

- Multilingual Zero-Shot Image Classification [\[see details\]](./clip_benchmark#multilingual-imagenet-1k)

  EN: English, ZH: Chinese, JP: Japanese, Ar: Arabic, IT: Italian

  | method            | IN-1K (EN) | IN-1K (ZH) | IN-1K (JP) | IN-1K (AR) | IN-1K (IT) |
  | ----------------- | :--------: | :--------: | :--------: | :--------: | :--------: |
  | Taiyi-CLIP-ViT-H  |     -      |    54.4    |     -      |     -      |     -      |
  | WuKong-ViT-L-G    |     -      |    57.5    |     -      |     -      |     -      |
  | CN-CLIP-ViT-H     |     -      |    59.6    |     -      |     -      |     -      |
  | AltCLIP-ViT-L     |    74.5    |    59.6    |     -      |     -      |     -      |
  | EVA-02-CLIP-E+    |    82.0    |     -      |     -      |     -      |    41.2    |
  | OpenCLIP-XLM-R-H  |    77.0    |    55.7    |    53.1    |    37.0    |    56.8    |
  | InternVL-C (ours) |    83.2    |    64.5    |    61.5    |    44.9    |    65.7    |

- Zero-Shot Video Classification \[see details\]

  | method            | #frame | K400 | K600 | K700 |
  | ----------------- | :----: | :--: | :--: | :--: |
  | OpenCLIP-G        |   1    | 65.9 | 66.1 | 59.2 |
  | EVA-02-CLIP-E+    |   1    | 69.8 | 69.3 | 63.4 |
  | InternVL-C (ours) |   1    | 71.0 | 71.3 | 65.7 |
  | ViCLIP            |   8    | 75.7 | 73.5 | 66.4 |
  | InternVL-C (ours) |   8    | 79.4 | 78.8 | 71.5 |

</details>

<details>
  <summary>Cross-Modal Retrieval (click to expand)</summary>

- English Zero-Shot Image-Text Retrieval [\[see details\]](./clip_benchmark#flickr30k--coco)

  <table>
    <tr  align=center>
        <td rowspan="3" align=left><b>model</b></td>
        <td colspan="6" align=center><b>Flickr30K</b></td>
        <td colspan="6" align=center><b>COCO</b></td>
        <td rowspan="3" align=center><b>avg</b></td>

  </tr>
     <tr  align=center>
        <td colspan="3" align=center><b>image-to-text</b></td>
        <td colspan="3" align=center><b>text-to-image</b></td>
         <td colspan="3" align=center><b>image-to-text</b></td>
        <td colspan="3" align=center><b>text-to-image</b></td>
     </tr>
     <tr>
        <td>R@1</td>
        <td>R@5</td>
        <td>R@10</td>
        <td>R@1</td>
        <td>R@5</td>
        <td>R@10</td>
        <td>R@1</td>
        <td>R@5</td>
        <td>R@10</td>
        <td>R@1</td>
        <td>R@5</td>
        <td>R@10</td>
     </tr>

  <tr align=center>
        <td align=left>OpenCLIP-G</td>
        <td>92.9</td>
        <td>99.3</td>
        <td>99.8</td>
        <td>79.5</td>
        <td>95.0</td>
        <td>97.1</td>
        <td>67.3</td>
        <td>86.9</td>
        <td>92.6</td>
        <td>51.4</td>
        <td>74.9</td>
        <td>83.0</td>
        <td>85.0</td>
     </tr>
  <tr align=center>
        <td align=left>EVA-02-CLIP-E+</td>
        <td>93.9</td>
        <td>99.4</td>
        <td>99.8</td>
        <td>78.8</td>
        <td>94.2</td>
        <td>96.8</td>
        <td>68.8</td>
        <td>87.8</td>
        <td>92.8</td>
        <td>51.1</td>
        <td>75.0</td>
        <td>82.7</td>
        <td>85.1</td>
     </tr>
    <tr align=center>
        <td align=left>EVA-CLIP-8B</td>
        <td>95.6</td>
        <td>99.6</td>
        <td>99.9</td>
        <td>80.8</td>
        <td>95.5</td>
        <td>97.6</td>
        <td>70.3</td>
        <td>89.3</td>
        <td>93.9</td>
        <td>53.0</td>
        <td>76.0</td>
        <td>83.4</td>
        <td>86.2</td>
     </tr>
  <tr align=center>
        <td align=left>InternVL-C (ours)</td>
        <td>94.7</td>
        <td>99.6</td>
        <td>99.9</td>
        <td>81.7</td>
        <td>96.0</td>
        <td>98.2</td>
        <td>70.6</td>
        <td>89.0</td>
        <td>93.5</td>
        <td>54.1</td>
        <td>77.3</td>
        <td>84.6</td>
        <td>86.6</td>
     </tr>
  <tr align=center>
        <td align=left>InternVL-G (ours)</td>
        <td>95.7</td>
        <td>99.7</td>
        <td>99.9</td>
        <td>85.0</td>
        <td>97.0</td>
        <td>98.6</td>
        <td>74.9</td>
        <td>91.3</td>
        <td>95.2</td>
        <td>58.6</td>
        <td>81.3</td>
        <td>88.0</td>
        <td>88.8</td>
     </tr>

  </table>

- Chinese Zero-Shot Image-Text Retrieval [\[see details\]](./clip_benchmark#flickr30k-cn--coco-cn)

  <table>
    <tr  align=center>
        <td rowspan="3" align=left><b>model</b></td>
        <td colspan="6" align=center><b>Flickr30K-CN</b></td>
        <td colspan="6" align=center><b>COCO-CN</b></td>
        <td rowspan="3" align=center><b>avg</b></td>

  </tr>
     <tr  align=center>
        <td colspan="3" align=center><b>image-to-text</b></td>
        <td colspan="3" align=center><b>text-to-image</b></td>
         <td colspan="3" align=center><b>image-to-text</b></td>
        <td colspan="3" align=center><b>text-to-image</b></td>
     </tr>
     <tr>
        <td>R@1</td>
        <td>R@5</td>
        <td>R@10</td>
        <td>R@1</td>
        <td>R@5</td>
        <td>R@10</td>
        <td>R@1</td>
        <td>R@5</td>
        <td>R@10</td>
        <td>R@1</td>
        <td>R@5</td>
        <td>R@10</td>
     </tr>

  <tr align=center>
        <td align=left>CN-CLIP-ViT-H</td>
        <td>81.6</td>
        <td>97.5</td>
        <td>98.8</td>
        <td>71.2</td>
        <td>91.4</td>
        <td>95.5</td>
        <td>63.0</td>
        <td>86.6</td>
        <td>92.9</td>
        <td>69.2</td>
        <td>89.9</td>
        <td>96.1</td>
        <td>86.1</td>
     </tr>

  <tr align=center>
        <td align=left>OpenCLIP-XLM-R-H</td>
        <td>86.1</td>
        <td>97.5</td>
        <td>99.2</td>
        <td>71.0</td>
        <td>90.5</td>
        <td>94.9</td>
        <td>70.0</td>
        <td>91.5</td>
        <td>97.0</td>
        <td>66.1</td>
        <td>90.8</td>
        <td>96.0</td>
        <td>87.6</td>
     </tr>

  <tr align=center>
        <td align=left>InternVL-C (ours)</td>
        <td>90.3</td>
        <td>98.8</td>
        <td>99.7</td>
        <td>75.1</td>
        <td>92.9</td>
        <td>96.4</td>
        <td>68.8</td>
        <td>92.0</td>
        <td>96.7</td>
        <td>68.9</td>
        <td>91.9</td>
        <td>96.5</td>
        <td>89.0</td>
     </tr>
  <tr align=center>
        <td align=left>InternVL-G (ours)</td>
        <td>92.9</td>
        <td>99.4</td>
        <td>99.8</td>
        <td>77.7</td>
        <td>94.8</td>
        <td>97.3</td>
        <td>71.4</td>
        <td>93.9</td>
        <td>97.7</td>
        <td>73.8</td>
        <td>94.4</td>
        <td>98.1</td>
        <td>90.9</td>
     </tr>

  </table>

- Multilingual Zero-Shot Image-Text Retrieval on XTD [\[see details\]](./clip_benchmark#xtd)

  | method            |  EN  |  ES  |  FR  |  ZH  |  IT  |  KO  |  RU  |  JP  | average |
  | ----------------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-----: |
  | AltCLIP           | 95.4 | 94.1 | 92.9 | 95.1 | 94.2 | 94.4 | 91.8 | 91.7 |  93.7   |
  | OpenCLIP-XLM-R-H  | 97.3 | 96.1 | 94.5 | 94.7 | 96.0 | 90.2 | 93.9 | 94.0 |  94.6   |
  | InternVL-C (ours) | 97.3 | 95.7 | 95.1 | 95.6 | 96.0 | 92.2 | 93.3 | 95.5 |  95.1   |
  | InternVL-G (ours) | 98.6 | 97.7 | 96.5 | 96.7 | 96.9 | 95.1 | 94.8 | 96.1 |  96.6   |

</details>

<details>
  <summary>Multimodal Dialogue (see "Compared with SOTA VLLMs")</summary>
</details>

## Quick Start with Huggingface

<details>
  <summary>using InternViT-6B (click to expand)</summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-6B-224px',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda().eval()

image = Image.open('./examples/image1.jpg').convert('RGB')

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-6B-224px')

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

outputs = model(pixel_values)
```

</details>

<details>
  <summary>using InternVL-C(ontrastive) and InternVL-G(enerative) (click to expand)</summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer


model = AutoModel.from_pretrained(
    'OpenGVLab/InternVL-14B-224px',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda().eval()

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternVL-14B-224px')

tokenizer = AutoTokenizer.from_pretrained(
    'OpenGVLab/InternVL-14B-224px', use_fast=False, add_eos_token=True)
tokenizer.pad_token_id = 0  # set pad_token_id to 0

images = [
    Image.open('./examples/image1.jpg').convert('RGB'),
    Image.open('./examples/image2.jpg').convert('RGB'),
    Image.open('./examples/image3.jpg').convert('RGB')
]
prefix = 'summarize:'
texts = [
    prefix + 'a photo of a red panda',  # English
    prefix + '一张熊猫的照片',  # Chinese
    prefix + '二匹の猫の写真'  # Japanese
]

pixel_values = image_processor(images=images, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()
input_ids = tokenizer(texts, return_tensors='pt', max_length=80,
                      truncation=True, padding='max_length').input_ids.cuda()

# InternVL-C
logits_per_image, logits_per_text = model(
    image=pixel_values, text=input_ids, mode='InternVL-C')
probs = logits_per_image.softmax(dim=-1)
# tensor([[9.9609e-01, 5.2185e-03, 6.0070e-08],
#         [2.2949e-02, 9.7656e-01, 5.9903e-06],
#         [3.2932e-06, 7.4863e-05, 1.0000e+00]], device='cuda:0',
#        dtype=torch.bfloat16, grad_fn=<SoftmaxBackward0>)

# InternVL-G
logits_per_image, logits_per_text = model(
    image=pixel_values, text=input_ids, mode='InternVL-G')
probs = logits_per_image.softmax(dim=-1)
# tensor([[9.9609e-01, 3.1738e-03, 3.6322e-08],
#         [8.6060e-03, 9.9219e-01, 2.8759e-06],
#         [1.7583e-06, 3.1233e-05, 1.0000e+00]], device='cuda:0',
#        dtype=torch.bfloat16, grad_fn=<SoftmaxBackward0>)

# please set add_eos_token to False for generation
tokenizer.add_eos_token = False
image = Image.open('./examples/image1.jpg').convert('RGB')
pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

tokenized = tokenizer("English caption:", return_tensors='pt')
pred = model.generate(
    pixel_values=pixel_values,
    input_ids=tokenized.input_ids.cuda(),
    attention_mask=tokenized.attention_mask.cuda(),
    num_beams=5,
    min_new_tokens=8,
)
caption = tokenizer.decode(pred[0].cpu(), skip_special_tokens=True).strip()
# English caption: a red panda sitting on top of a wooden platform
```

</details>

<details>
  <summary>using InternVL-Chat (click to expand)</summary>

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


path = "OpenGVLab/InternVL-Chat-V1-5"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
# Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

# single-round single-image conversation
question = "请详细描述图片" # Please describe the picture in detail
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(question, response)

# multi-round single-image conversation
question = "请详细描述图片" # Please describe the picture in detail
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(question, response)

question = "请根据图片写一首诗" # Please write a poem according to the picture
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(question, response)

# multi-round multi-image conversation
pixel_values1 = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

question = "详细描述这两张图片" # Describe the two pictures in detail
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(question, response)

question = "这两张图片的相同点和区别分别是什么" # What are the similarities and differences between these two pictures
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(question, response)

# batch inference (single image per sample)
pixel_values1 = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
image_counts = [pixel_values1.size(0), pixel_values2.size(0)]
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

questions = ["Describe the image in detail."] * len(image_counts)
responses = model.batch_chat(tokenizer, pixel_values,
                             image_counts=image_counts,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(question)
    print(response)
```

</details>

## Inference Acceleration by LMDeploy

We recommend using [LMDeploy](https://github.com/InternLM/lmdeploy), if InternVL-Chat model inference optimization is required.

In the following subsections, we will introduce the usage of LMDeploy with the [InternVL-Chat-V1-5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) model as an example.

First of all, please setup the inference environment as follows:

```shell
conda create -n internvl python=3.10 -y
conda activate internvl

pip install timm torchvision==0.17.2
pip install lmdeploy
```

LMDeploy pypi package depends on CUDA 12.x by default. For a CUDA 11.x environment, please refer to the [installation guide](https://lmdeploy.readthedocs.io/en/latest/get_started.html#installation).

### Offline Inference Pipeline

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image
pipe = pipeline('OpenGVLab/InternVL-Chat-V1-5')
image = load_image('examples/image2.jpg')
response = pipe(('describe this image', image))
print(response)
```

For more on using the VLM pipeline, including multi-image inference or multi-turn chat, please overview [this](https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html) guide.

### Online Inference Service

LMDeploy supports one-click packaging of the VLM model into an OpenAI service, providing seamless integration with the OpenAI API.

The service can be launched by one command as below:

```shell
lmdeploy serve api_server OpenGVLab/InternVL-Chat-V1-5
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.

For more details, including service startup with docker, RESTful API information, and openai integration methods, please refer to [this](https://lmdeploy.readthedocs.io/en/latest/serving/api_server_vl.html) guide.

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider cite:

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

## Acknowledgement

InternVL is built with reference to the code of the following projects: [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip), [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), [EVA](https://github.com/baaivision/EVA/tree/master), [InternImage](https://github.com/OpenGVLab/InternImage), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Transformers](https://github.com/huggingface/transformers), [DINOv2](https://github.com/facebookresearch/dinov2), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm), and [LLaVA-1.5](https://github.com/haotian-liu/LLaVA). Thanks for their awesome work!

______________________________________________________________________

If you want to join our WeChat group, please scan the following QR Code to add our assistant as a Wechat friend:

<p align="center"><img width="300" alt="image" src="https://github.com/OpenGVLab/DragGAN/assets/26198430/e3f0807f-956a-474e-8fd2-1f7c22d73997"></p>
