# <img width="60" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/47669167/7037290e-f474-4d11-b90f-1d8316087bf8"> InternVL家族：通过开源组件缩小与商业多模态模型的差距 —— GPT-4V的开源替代方案

[\[📖 英文版本\]](./README.md) [\[🆕 博客\]](https://internvl.github.io/blog/)  [\[📜 InternVL 1.0 论文\]](https://arxiv.org/abs/2312.14238)  [\[📜 InternVL 1.5 技术报告\]](https://arxiv.org/abs/2404.16821)  [\[🗨️ Chat Demo\]](https://internvl.opengvlab.com/) [\[🤗 HuggingFace Demo\]](https://huggingface.co/spaces/OpenGVLab/InternVL)

[\[🚀 快速开始\]](#使用-huggingface-快速开始)  [\[🌐 Community-hosted API\]](https://rapidapi.com/adushar1320/api/internvl-chat)  [\[📖 中文解读\]](https://zhuanlan.zhihu.com/p/675877376)

<a href="https://trendshift.io/repositories/9803" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9803" alt="OpenGVLab%2FInternVL | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

## 最新消息🚀🚀🚀
- `2024/06/19`: 🚀 我们提出了 Needle In A Multimodal Haystack ([MM-NIAH](https://github.com/OpenGVLab/MM-NIAH))，这是第一个针对模型关于长多模态文档理解能力的评测基准。**实验结果表明，Gemini-1.5在包含图像针的数据上的性能并不比乱猜的性能要好。**
- `2024/06/04`: InternVL 1.5 在 [Video-MME](https://github.com/BradyFU/Video-MME) 数据集的 Image MLLM 类别中实现了SOTA的性能，展示了在多图场景下的泛化能力，超过了许多专门的 Video MLLM，并接近开源SOTA视频模型 LLaVA-Next-Video。
- `2024/05/29`: 🚀 我们开源了 Mini-InternVL-Chat 系列，目前包括以下两个模型：[Mini-InternVL-Chat-2B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5) 和 [Mini-InternVL-Chat-4B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5)。我们的小模型在极小的尺寸下实现了令人印象深刻的性能：2B模型仅以8%的模型尺寸实现了80%的性能，4B模型以16%的模型尺寸实现了90%的性能。更多细节请查看我们的[博客](https://internvl.github.io/blog/2024-05-25-Mini-InternVL-1.5/)。
- `2024/05/28`: 感谢 [lmdeploy](https://github.com/InternLM/lmdeploy) 团队提供的AWQ量化支持。4-bit模型发布在 [OpenGVLab/InternVL-Chat-V1-5-AWQ](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-AWQ)。
- `2024/05/13`: 🔥 InternVL 现在可以作为扩散模型的 [文本编码器](https://huggingface.co/OpenGVLab/InternVL-14B-224px)，支持全球超过110种语言的多语言生成。详情请看 [MuLan](https://github.com/mulanai/MuLan)。
- `2024/04/28`: 我们发布了 InternVL-Chat-V1-5 的 INT8 量化版本，详细请看 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-Int8)。
- `2024/04/28`: 我们在 Infographics VQA 的基准测试中达到了 SOTA 性能（75.74），详情请看 [here](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=3)。
- `2024/04/18`: InternVL-Chat-V1-5 已经在 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) 发布，在MMMU、DocVQA、ChartQA、MathVista等各种基准测试中，性能接近GPT-4V和Gemini Pro。
- `2024/02/27`: InternVL 被 CVPR 2024 接收！🎉
- `2024/02/24`: InternVL-Chat 模型已经接入 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)。
- `2024/02/21`: [InternVL-Chat-V1-2-Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) 在 MathVista（59.9）、MMBench（83.8）和MMVP（58.7）上达到了SOTA性能。详情请参见我们的 [blog](<[BLOG.md](https://internvl.github.io/blog/2024-02-21-InternVL-1.2/)>)。
- `2024/02/12`: InternVL-Chat-V1-2 已经发布。它在MMMU验证集上达到了51.6的分数，在MMBench测试集上达到了82.3的分数。 更多信息请参考 [blog](<[BLOG.md](https://internvl.github.io/blog/2024-02-21-InternVL-1.2/)>)、[SFT data](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets) 或者尝试我们的 [demo](https://internvl.opengvlab.com/)。该模型已经在 [HuggingFace](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2) 发布，训练、测评的数据和脚本均已开源。
- `2024/02/04`: [InternVL-Chat-V1-1](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1) 在 [MMVP](https://github.com/tsb0601/MMVP) 上达到了 44.67 的得分，高于GPT-4V！
- `2024/01/27`: 我们发布了448分辨率的模型，在MMBench的验证集上达到了76.6的分数，详情请看 [here](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#-evaluation-chinese-models)。
- `2024/01/24`: InternVL-Chat-V1-1 已经发布，它支持中文，并且有强大的OCR能力，详情请看 [here](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1) 或者尝试我们的 [demo](https://internvl.opengvlab.com/)。
- `2024/01/16`: 我们发布了 [定制化的 mmcv/mmsegmentation/mmdetection code](https://github.com/OpenGVLab/InternVL-MMDetSeg)，集成了DeepSpeed，可以用于训练目标检测和语义分割大模型。

## 文档

- 安装

  - 如何搭建运行环境?  [\[link\]](./INSTALLATION.md)

- 训练或者微调

  - 如何复现 InternVL-Chat-V1-2 的SFT阶段? [\[link\]](./internvl_chat#start-training)
  - 如何在自定义数据集上微调 InternVL-Chat-V1-2? [\[link\]](./document/how_to_finetune_internvl_chat_v1_2_on_a_custom_dataset.md)
  - 如何在自定义数据集上微调 Mini-InternVL-Chat 系列? [\[link\]](./document/How_to_finetune_mini_internvl_chat_v1_5_on_a_custom_dataset.md)

- Benchmark 测评

  > 由于此代码库与 VLMEvalKit 之间存在细微的实现差异，在测试同一模型时，性能指标可能会出现轻微差异。

  - 如何评测 InternVL-Chat-V1-5? [\[link\]](./document/how_to_evaluate_internvl_chat_v1_5.md)
  - 如何使用 VLMEvalKit 评测 InternVL-Chat-V1-5? (推荐) [\[link\]](./document/how_to_evaluate_internvl_chat_v1_5_using_vlmevalkit.md)
  - 如何使用 VLMEvalKit 评测 Mini-InternVL-Chat-2B-V1-5? (推荐) [\[link\]](./document/how_to_evaluate_mini_internvl_chat_2b_v1_5_using_vlmevalkit.md)
  - 如何使用 VLMEvalKit 评测 Mini-InternVL-Chat-4B-V1-5? (推荐) [\[link\]](./document/how_to_evaluate_mini_internvl_chat_4b_v1_5_using_vlmevalkit.md)

- 部署

  - 如何部署本地的 demo? [\[link\]](./document/how_to_deploy_a_local_demo.md)
  - 如何用 Nvidia V100 GPU 运行 InternVL-1.5 8bit? [\[link\]](https://github.com/OpenGVLab/InternVL/issues/144) [\[中文教程\]](https://zhuanlan.zhihu.com/p/697188143)
  - 如何进行批量推理？ [\[link\]](https://github.com/OpenGVLab/InternVL/blob/main/README.md?plain=1#L617)
  - LMDeploy 加速推理 [\[link\]](#inference-acceleration-by-lmdeploy) [\[中文教程\]](https://zhuanlan.zhihu.com/p/696955211)

## 和 SOTA 多模态大模型对比

<p align="center"><img width="500" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/38e8a632-229c-4b20-b7e1-77299dfc6cee"></p>

<img width="1229" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/e9065a58-86fa-47ef-be9a-eb734532e73f">

<img width="1229" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/2b4f2978-36ea-4065-841d-3651c58955ed">

## 什么是 InternVL?

InternVL 将 ViT 拓展到 _**6B 参数**_ 并与大语言模型对齐。

## 模型

**多模态大语言模型**

| Model                      | Date       | Download                                                                             | Note                                                                                                             |
| -------------------------- | ---------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Mini&#8209;InternVL&#8209;Chat&#8209;4B&#8209;V1&#8209;5 | 2024.05.28 | 🤗 [HF link](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5)            | 🚀🚀 16% 的模型大小，90% 的模型性能                                                                              |
| Mini-InternVL-Chat-2B-V1-5 | 2024.05.19 | 🤗 [HF link](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5)            | 🚀🚀 8% 的模型大小，80% 的模型性能                                                                               |
| InternVL-Chat-V1-5-AWQ     | 2024.05.28 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-AWQ)                | InternVL-Chat-V1-5的 INT4 版本                                                                                   |
| InternVL-Chat-V1-5-Int8    | 2024.04.28 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-Int8)               | InternVL-Chat-V1-5的 INT8 版本                                                                                   |
| InternVL-Chat-V1-5         | 2024.04.18 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)                    | 支持4K图像；超强OCR性能；在MMMU、DocVQA、ChartQA、MathVista等各种基准测试中，其性能接近GPT-4V和Gemini Pro (🔥新) |
| InternVL-Chat-V1-2-Plus    | 2024.02.21 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus)               | 更多的SFT数据并且更强大                                                                                          |
| InternVL-Chat-V1-2         | 2024.02.11 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2)                    | 拓展 LLM 到 34B                                                                                                  |
| InternVL-Chat-V1-1         | 2024.01.24 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1)                    | 支持中文并且有强大的OCR能力                                                                                      |
| InternVL-Chat-19B-448px    | 2024.02.03 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B-448px) | 448 分辨率                                                                                                       |
| InternVL-Chat-19B          | 2023.12.25 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B)       | 英语多模态对话大模型                                                                                             |
| InternVL-Chat-13B          | 2023.12.25 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B)        | 英语多模态对话大模型                                                                                             |

**视觉语言基础模型**

| Model                   | Date       | Download                                                               | Note                                                            |
| ----------------------- | ---------- | ---------------------------------------------------------------------- | --------------------------------------------------------------- |
| InternViT-300M-448px    | 2024.05.25 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-300M-448px)    | 蒸馏的300M小型视觉基础模型 (🔥新)                               |
| InternViT-6B-448px-V1-5 | 2024.04.20 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) | 支持动态分辨率，十分强大的OCR能力 (🔥新)                        |
| InternViT-6B-448px-V1-2 | 2024.02.11 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2) | 448 分辨率                                                      |
| InternViT&#8209;6B&#8209;448px&#8209;V1&#8209;0 | 2024.01.30 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-0) | 448 分辨率                                                      |
| InternViT-6B-224px      | 2023.12.22 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-224px)      | 视觉基础模型                                                    |
| InternVL-14B-224px      | 2023.12.22 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-14B-224px)      | 视觉语言基础模型，InternViT-6B + QLLaMA，可以用于做图文对的检索 |

## InternVL 可以做什么?

<details>
  <summary>视觉感知 (点击展开)</summary>

- Linear-Probe 图像分类 [\[see details\]](./classification#-evaluation)

  ViT-22B uses the private JFT-3B dataset.

  | method              | #param | IN-1K | IN-ReaL | IN-V2 | IN-A | IN-R | IN-Sketch |
  | ------------------- | :----: | :---: | :-----: | :---: | :--: | :--: | :-------: |
  | OpenCLIP-G          |  1.8B  | 86.2  |  89.4   | 77.2  | 63.8 | 87.8 |   66.4    |
  | DINOv2-g            |  1.1B  | 86.5  |  89.6   | 78.4  | 75.9 | 78.8 |   62.5    |
  | EVA-01-CLIP-g       |  1.1B  | 86.5  |  89.3   | 77.4  | 70.5 | 87.7 |   63.1    |
  | MAWS-ViT-6.5B       |  6.5B  | 87.8  |    -    |   -   |  -   |  -   |     -     |
  | ViT-22B\*           | 21.7B  | 89.5  |  90.9   | 83.2  | 83.8 | 87.4 |     -     |
  | InternViT-6B (ours) |  5.9B  | 88.2  |  90.4   | 79.9  | 77.5 | 89.8 |   69.1    |

- 语义分割 [\[see details\]](./segmentation#-evaluation)

  | method                | decoder | #param (train/total) | crop size | mIoU         |
  | --------------------- | :-----: | :------------------: | :-------: | ------------ |
  | OpenCLIP-G (frozen)   | Linear  |     0.3M / 1.8B      |    512    | 39.3         |
  | ViT-22B (frozen)      | Linear  |     0.9M / 21.7B     |    504    | 34.6         |
  | InternViT-6B (frozen) | Linear  |     0.5M / 5.9B      |    504    | 47.2 (+12.6) |
  | ViT-22B (frozen)      | UperNet |     0.8B / 22.5B     |    504    | 52.7         |
  | InternViT-6B (frozen) | UperNet |     0.4B / 6.3B      |    504    | 54.9 (+2.2)  |
  | ViT-22B               | UperNet |    22.5B / 22.5B     |    504    | 55.3         |
  | InternViT-6B          | UperNet |     6.3B / 6.3B      |    504    | 58.9 (+3.6)  |

- 零样本图像分类 [\[see details\]](./clip_benchmark#imagenet-variants-and-objectnet)

  | method            | IN-1K | IN-A | IN-R | IN-V2 | IN-Sketch | ObjectNet |
  | ----------------- | :---: | :--: | :--: | :---: | :-------: | :-------: |
  | OpenCLIP-G        | 80.1  | 69.3 | 92.1 | 73.6  |   68.9    |   73.0    |
  | EVA-02-CLIP-E+    | 82.0  | 82.1 | 94.5 | 75.7  |   71.6    |   79.6    |
  | ViT-22B\*         | 85.9  | 90.1 | 96.0 | 80.9  |     -     |   87.6    |
  | InternVL-C (ours) | 83.2  | 83.8 | 95.5 | 77.3  |   73.9    |   80.6    |

- 多语言零样本图像分类 [\[see details\]](./clip_benchmark#multilingual-imagenet-1k)

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

- 零样本视频分类 \[see details\]

  | method            | #frame | K400 | K600 | K700 |
  | ----------------- | :----: | :--: | :--: | :--: |
  | OpenCLIP-G        |   1    | 65.9 | 66.1 | 59.2 |
  | EVA-02-CLIP-E+    |   1    | 69.8 | 69.3 | 63.4 |
  | InternVL-C (ours) |   1    | 71.0 | 71.3 | 65.7 |
  | ViCLIP            |   8    | 75.7 | 73.5 | 66.4 |
  | InternVL-C (ours) |   8    | 79.4 | 78.8 | 71.5 |

</details>

<details>
  <summary>跨模态检索 (点击展开)</summary>

- 英语零样本图文检索 [\[see details\]](./clip_benchmark#flickr30k--coco)

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

- 中文零样本图文对检索 [\[see details\]](./clip_benchmark#flickr30k-cn--coco-cn)

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

- 多语言零样本图文对检索 [\[see details\]](./clip_benchmark#xtd)

  | method            |  EN  |  ES  |  FR  |  ZH  |  IT  |  KO  |  RU  |  JP  | average |
  | ----------------- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :-----: |
  | AltCLIP           | 95.4 | 94.1 | 92.9 | 95.1 | 94.2 | 94.4 | 91.8 | 91.7 |  93.7   |
  | OpenCLIP-XLM-R-H  | 97.3 | 96.1 | 94.5 | 94.7 | 96.0 | 90.2 | 93.9 | 94.0 |  94.6   |
  | InternVL-C (ours) | 97.3 | 95.7 | 95.1 | 95.6 | 96.0 | 92.2 | 93.3 | 95.5 |  95.1   |
  | InternVL-G (ours) | 98.6 | 97.7 | 96.5 | 96.7 | 96.9 | 95.1 | 94.8 | 96.1 |  96.6   |

</details>

<details>
  <summary>多模态对话 (请看 "和SOTA的多模态大模型对比")</summary>
</details>

## 使用 Huggingface 快速开始

<details>
  <summary>使用 InternViT-6B (点击展开)</summary>

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
  <summary>使用 InternVL-C(ontrastive) 和 InternVL-G(enerative) (点击展开)</summary>

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
  <summary>使用 InternVL-Chat (点击展开)</summary>

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

## 通过 LMDeploy 加速推理

如果需要优化InternVL-Chat模型的推理，我们推荐使用 [LMDeploy](https://github.com/InternLM/lmdeploy)。

在接下来的小节中，我们将以 [InternVL-Chat-V1-5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) 模型为例介绍 LMDeploy 的使用

首先，请按照下面的步骤设置推理环境:

```shell
conda create -n internvl python=3.10 -y
conda activate internvl

pip install timm torchvision==0.17.2
pip install lmdeploy
```

LMDeploy 的 pypi 包默认依赖 CUDA 12.x。对于 CUDA 11.x 环境，请参考  [installation guide](https://lmdeploy.readthedocs.io/en/latest/get_started.html#installation).

### 离线推理过程

```python
from lmdeploy import pipeline
from lmdeploy.vl import load_image
pipe = pipeline('OpenGVLab/InternVL-Chat-V1-5')
image = load_image('examples/image2.jpg')
response = pipe(('describe this image', image))
print(response)
```

有关使用VLM流程的更多信息，包括图像推理或多轮对话，请查看指南 [guide](https://lmdeploy.readthedocs.io/en/latest/inference/vl_pipeline.html) 。

### 在线推理服务

LMDeploy支持将VLM模型一键打包成OpenAI服务，实现与OpenAI API的无缝集成。

该服务可以通过以下命令启动：

```shell
lmdeploy serve api_server OpenGVLab/InternVL-Chat-V1-5
```

`api_server`的参数可以通过命令`lmdeploy serve api_server -h`查看，例如，使用`--tp`设置张量并行度，使用`--session-len`指定上下文窗口的最大长度，使用`--cache-max-entry-count`调整用于k/v缓存的GPU内存比例等。

有关更多详细信息，包括使用Docker启动服务、RESTful API信息以及OpenAI集成方法，请查看指导 [guide](https://lmdeploy.readthedocs.io/en/latest/serving/api_server_vl.html)。

## 许可证

本项目遵循[MIT license](LICENSE)许可证发布。项目中的部分代码和模型来自其他来源，并受其各自许可证的约束。

## 引用

如果您在研究中发现本项目有用，请考虑引用：

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

## 感谢

InternVL 的代码构建参考了以下项目: [OpenAI CLIP](https://github.com/openai/CLIP)、[Open CLIP](https://github.com/mlfoundations/open_clip)、[CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)、[EVA](https://github.com/baaivision/EVA/tree/master)、[InternImage](https://github.com/OpenGVLab/InternImage)、[ViT-Adapter](https://github.com/czczup/ViT-Adapter)、[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)、[Transformers](https://github.com/huggingface/transformers)、[DINOv2](https://github.com/facebookresearch/dinov2)、[BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)、[Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm)和 [LLaVA-1.5](https://github.com/haotian-liu/LLaVA)。感谢他们的工作。

______________________________________________________________________

如何你想加入我们的项目群，请扫描下方二维码添加我们的小助手。

<p align="center"><img width="300" alt="image" src="https://github.com/OpenGVLab/DragGAN/assets/26198430/e3f0807f-956a-474e-8fd2-1f7c22d73997"></p>
