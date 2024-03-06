# <img width="60" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/8529570/5aa4cda8-b453-40a0-9336-17012b430ae8"> InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks —— An Open-Source Alternative to ViT-22B

\[[Update Blog](./BLOG.md)\]   \[[Paper](https://arxiv.org/abs/2312.14238)\]  \[[Chat Demo](https://internvl.opengvlab.com/)\]  \[[Quick Start](#quick-start-with-huggingface)\]  \[[中文解读](https://mp.weixin.qq.com/s/bdfAJRqOF9tUk8Vy9KC_XQ)\]

## News🚀🚀🚀

- `2024/02/27`: InternVL is accepted by CVPR 2024! 🎉
- `2024/02/24`: InternVL-Chat models have been included in the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
- `2024/02/21`: [InternVL-Chat-V1.2-Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2-Plus) achieves SOTA performance on MathVista (59.9), MMBench (83.8), and MMVP (58.7). See our [blog](BLOG.md) for more details.
- `2024/02/12`: InternVL-Chat-V1.2 has been released, utilizing [Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) as the LLM. It achieves 51.6 on MMMU val and 82.3 on MMBench test. For more details, please refer to our [blog](BLOG.md), [SFT data](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets) or try our [demo](https://internvl.opengvlab.com/). The model is now available on [HuggingFace](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2), and both training/evaluation data and scripts are open-sourced.
- `2024/02/04`: [InternVL-Chat-V1.1](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-1) achieves 44.67% on [MMVP](https://github.com/tsb0601/MMVP), higher than GPT-4V!
- `2024/01/27`: We release 448 resolution model, achieving 76.6 on MMBench dev, see [here](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#-evaluation-chinese-models).
- `2024/01/24`: InternVL-Chat-V1.1 is released, it supports Chinese and has stronger OCR capability, see [here](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-1) or try our [demo](https://internvl.opengvlab.com/).
- `2024/01/16`: We release our [customized mmcv/mmsegmentation/mmdetection code](https://github.com/OpenGVLab/InternVL-MMDetSeg), integrated with DeepSpeed, which can be used for training large-scale object detection and semantic segmentation models.

## What is InternVL?

InternVL scales up the ViT to _**6B parameters**_ and aligns it with LLM.

## Model Zoo

**Vision-Language Foundation Model**

| Model                   | Date       | Download                                                               | Note                             |
| ----------------------- | ---------- | ---------------------------------------------------------------------- | -------------------------------- |
| InternViT-6B-224px      | 2023.12.22 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-224px)      | vision foundation model          |
| InternVL-14B-224px      | 2023.12.22 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-14B-224px)      | vision-language foundation model |
| InternViT-6B-448px      | 2024.01.30 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px)      | 448 resolution                   |
| InternViT-6B-448px-V1.2 | 2024.02.11 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2) | 448 resolution (🔥new)           |

**Vision Large Language Model**

| Model                   | Date       | Download                                                                             | Note                               |
| ----------------------- | ---------- | ------------------------------------------------------------------------------------ | ---------------------------------- |
| InternVL-Chat-13B       | 2023.12.25 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B)        | English multimodal dialogue        |
| InternVL-Chat-19B       | 2023.12.25 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B)       | English multimodal dialogue        |
| InternVL-Chat-19B-448px | 2024.02.03 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B-448px) | 448 resolution                     |
| InternVL-Chat-V1.1      | 2024.01.24 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-1)            | support Chinese and stronger OCR   |
| InternVL-Chat-V1.2      | 2024.02.11 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2)            | scaling up LLM to 34B (🔥new)      |
| InternVL-Chat-V1.2-Plus | 2024.02.21 | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-Chinese-V1-2-Plus)       | more SFT data and stronger (🔥new) |

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
  | ViT-22B\*           | 21.7B  | 89.5  |  90.9   | 83.2  | 83.8 | 87.4 |     −     |
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
  | ViT-22B\*         | 85.9  | 90.1 | 96.0 | 80.9  |     −     |   87.6    |
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
  <summary>Multimodal Dialogue (click to expand)</summary>

- Compared with SOTA VLLMs

  | name               | image size | MMMU<br>(val) | MMMU<br>(test) | MathVista<br>(testmini) | MMB<br>(test) | MMB−CN<br>(test) | MMVP | MME      | ScienceQA<br>(image) | POPE | TextVQA | SEEDv1<br>(image) | VizWiz<br>(test) | GQA<br>(test) |
  | ------------------ | ---------- | ------------- | -------------- | ----------------------- | ------------- | ---------------- | ---- | -------- | -------------------- | ---- | ------- | ----------------- | ---------------- | ------------- |
  | GPT-4V\*           | unknown    | 56.8          | 55.7           | 49.9                    | 77.0          | 74.4             | 38.7 | 1409/517 | -                    | -    | 78.0    | 71.6              | -                | -             |
  | Gemini Ultra\*     | unknown    | 59.4          | -              | 53.0                    | -             | -                | -    | -        | -                    | -    | 82.3    | -                 | -                | -             |
  | Gemini Pro\*       | unknown    | 47.9          | -              | 45.2                    | 73.6          | 74.3             | 40.7 | 1497/437 | -                    | -    | 74.6    | 70.7              | -                | -             |
  | Qwen-VL-Plus\*     | unknown    | 45.2          | 40.8           | 43.3                    | 67.0          | 70.7             | -    | 1681/502 | -                    | -    | 78.9    | 65.7              | -                | -             |
  | Qwen-VL-Max\*      | unknown    | 51.4          | 46.8           | 51.0                    | 77.6          | 75.7             | -    | -        | -                    | -    | 79.5    | -                 | -                | -             |
  |                    |            |               |                |                         |               |                  |      |          |                      |      |         |                   |                  |               |
  | LLaVA-NEXT-34B     | 672x672    | 51.1          | 44.7           | 46.5                    | 79.3          | 79.0             | -    | 1631/397 | 81.8                 | 87.7 | 69.5    | 75.9              | 63.8             | 67.1          |
  | InternVL-Chat-V1.2 | 448x448    | 51.6          | 46.2           | 47.7                    | 82.2          | 81.2             | 56.7 | 1672/509 | 83.3                 | 88.0 | 69.7    | 75.6              | 60.0             | 64.0          |

  \* denotes proprietary models. MMBench results are collected from the [leaderboard](https://mmbench.opencompass.org.cn/leaderboard). In most benchmarks, InternVL-Chat-V1.2 achieves better performance than LLaVA-NeXT-34B.

- Zero-Shot Image Captioning [\[see details\]](./internvl_g#zero-shot-image-captioning)

  | method            | COCO  | Flickr30K | NoCaps |
  | ----------------- | :---: | :-------: | :----: |
  | Emu-I             | 117.7 |     -     |   -    |
  | DreamLLM          | 115.4 |     -     |   -    |
  | InternVL-G (ours) | 128.2 |   79.2    | 113.7  |

- Multimodal Benchmarks with Frozen LLM [\[see details\]](./internvl_chat#-evaluation)

  | method               | visual encoder | glue layer |  LLM  | res. | COCO  | Flickr | NoCaps | VQAv2 | GQA  | VizWiz | TextVQA |  MME   | POPE |
  | -------------------- | :------------: | :--------: | :---: | :--: | :---: | :----: | :----: | :---: | :--: | :----: | :-----: | :----: | :--: |
  | InstructBLIP         |     EVA-g      |  QFormer   | V-7B  | 224  |   –   |  82.4  | 123.1  |   –   | 49.2 |  34.5  |  50.1   |   –    |  –   |
  | BLIP-2               |     EVA-g      |  QFormer   | V-13B | 224  |   –   |  71.6  | 103.9  | 41.0  | 41.0 |  19.6  |  42.5   | 1293.8 | 85.3 |
  | InstructBLIP         |     EVA-g      |  QFormer   | V-13B | 224  |   –   |  82.8  | 121.9  |   –   | 49.5 |  33.4  |  50.7   | 1212.8 | 78.9 |
  | InternVL-Chat (ours) |    IViT-6B     |   QLLaMA   | V-7B  | 224  | 141.4 |  89.7  | 120.5  | 72.3  | 57.7 |  44.5  |  42.1   | 1298.5 | 85.2 |
  | InternVL-Chat (ours) |    IViT-6B     |   QLLaMA   | V-13B | 224  | 142.4 |  89.9  | 123.1  | 71.7  | 59.5 |  54.0  |  49.1   | 1317.2 | 85.4 |

- Multimodal Benchmarks with Trainable LLM [\[see details\]](./internvl_chat_llava)

  | method               | vision encoder |  LLM  | res. | VQAv2 | GQA  | VizWiz | SQA  | TextVQA | POPE |  MME   | MMB  | MMB<sub>CN</sub> | MMVet |
  | -------------------- | :------------: | :---: | :--: | :---: | :--: | :----: | :--: | :-----: | :--: | :----: | :--: | :--------------: | :---: |
  | LLaVA-1.5            |  CLIP-L-336px  | V-7B  | 336  | 78.5  | 62.0 |  50.0  | 66.8 |  58.2   | 85.9 | 1510.7 | 64.3 |       58.3       | 30.5  |
  | LLaVA-1.5            |  CLIP-L-336px  | V-13B | 336  | 80.0  | 63.3 |  53.6  | 71.6 |  61.3   | 85.9 | 1531.3 | 67.7 |       63.6       | 35.4  |
  | InternVL-Chat (ours) | IViT-6B-224px  | V-7B  | 336  | 79.3  | 62.9 |  52.5  | 66.2 |  57.0   | 86.4 | 1525.1 | 64.6 |       57.6       | 31.2  |
  | InternVL-Chat (ours) | IViT-6B-224px  | V-13B | 336  | 80.2  | 63.9 |  54.6  | 70.1 |  58.7   | 87.1 | 1546.9 | 66.5 |       61.9       | 33.7  |
  | InternVL-Chat (ours) | IViT-6B-448px  | V-13B | 448  | 82.0  | 64.1 |  60.1  | 71.6 |  64.8   | 87.2 | 1579.0 | 68.2 |       64.0       | 36.7  |

- Tiny LVLM [\[see details\]](https://github.com/OpenGVLab/Multi-Modality-Arena/tree/main/tiny_lvlm_evaluation)

  | Rank |                                        Model                                        |         Version          |   Score    |
  | :--: | :---------------------------------------------------------------------------------: | :----------------------: | :--------: |
  | 🏅️  |                **[InternVL](https://github.com/OpenGVLab/InternVL)**                |      InternVL-Chat       | **327.61** |
  |  🥈  |     **[InternLM-XComposer-VL](https://github.com/InternLM/InternLM-XComposer)**     | InternLM-XComposer-VL-7B | **322.51** |
  |  🥉  |                        **[Bard](https://bard.google.com/)**                         |           Bard           | **319.59** |
  |  4   |                  [Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL)                  |       Qwen-VL-Chat       |   316.81   |
  |  5   |                  [LLaVA-1.5](https://github.com/haotian-liu/LLaVA)                  |        Vicuna-7B         |   307.17   |
  |  6   | [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) |        Vicuna-7B         |   300.64   |
  |  7   |        [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)         |  InternLM-XComposer-7B   |   288.89   |
  |  8   |        [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)        |         FlanT5xl         |   284.72   |
  |  9   |                     [BLIVA](https://github.com/mlpc-ucsd/BLIVA)                     |        Vicuna-7B         |   284.17   |
  |  10  |                    [Lynx](https://github.com/bytedance/lynx-llm)                    |        Vicuna-7B         |   279.24   |

</details>

## Installation

See [INSTALLATION.md](./INSTALLATION.md)

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

- Single GPU

```python
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer

path = "OpenGVLab/InternVL-Chat-Chinese-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path)
image = Image.open('./examples/image2.jpg').convert('RGB')
image = image.resize((448, 448))
image_processor = CLIPImageProcessor.from_pretrained(path)

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

question = "请详细描述图片"
response = model.chat(tokenizer, pixel_values, question, generation_config)
```

- Multiple GPUs

```python
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer

path = "OpenGVLab/InternVL-Chat-Chinese-V1-1"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(path)
image = Image.open('./examples/image2.jpg').convert('RGB')
image = image.resize((448, 448))
image_processor = CLIPImageProcessor.from_pretrained(path)

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

question = "请详细描述图片"
response = model.chat(tokenizer, pixel_values, question, generation_config)
```

</details>

## Schedule

- [x] Release high-resolution models
- [x] Release InternVL-Chat
- [x] Release InternVL-C(ontrastive) and InternVL-G(enerative)
- [x] Release InternViT-6B

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
```

## Acknowledgement

InternVL is built with reference to the code of the following projects: [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip), [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), [EVA](https://github.com/baaivision/EVA/tree/master), [InternImage](https://github.com/OpenGVLab/InternImage), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Transformers](https://github.com/huggingface/transformers), [DINOv2](https://github.com/facebookresearch/dinov2), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm), and [LLaVA-1.5](https://github.com/haotian-liu/LLaVA). Thanks for their awesome work!

______________________________________________________________________

If you want to join our WeChat group, please scan the following QR Code to add our assistant as a Wechat friend:

<p align="center"><img width="300" alt="image" src="https://github.com/OpenGVLab/DragGAN/assets/26198430/e3f0807f-956a-474e-8fd2-1f7c22d73997"></p>
