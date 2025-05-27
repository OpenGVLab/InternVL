<div align="center">

# InternVL Family: Closing the Gap to Commercial Multimodal Models with Open-Source Suites —— A Pioneering Open-Source Alternative to GPT-4o

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/930e6814-8a9f-43e1-a284-118a5732daa4">
  <br>
</div>

[\[🆕 Blog\]](https://internvl.github.io/blog/)  [\[🤔 FAQs\]](https://internvl.readthedocs.io/en/latest/tutorials/faqs.html)   [\[🗨️ Chat Demo\]](https://internvl.opengvlab.com/)  [\[🤗 HF Demo\]](https://huggingface.co/spaces/OpenGVLab/InternVL)  [\[📖 Document\]](https://internvl.readthedocs.io/en/latest/)  [\[🌐 API\]](https://internlm.intern-ai.org.cn/api/document)  [\[🚀 Quick Start\]](#quick-start-with-huggingface)

[\[🔥 InternVL3.0 Report\]](https://huggingface.co/papers/2504.10479) [\[🔥 InternVL2.5 MPO\]](https://huggingface.co/papers/2411.10442) [\[🔥 InternVL2.5 Report\]](https://huggingface.co/papers/2412.05271) [\[Mini-InternVL Paper\]](https://arxiv.org/abs/2410.16261) [\[InternVL2 Blog\]](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)  [\[📜 InternVL 1.5 Paper\]](https://huggingface.co/papers/2404.16821)  [\[📜 InternVL 1.0 Paper\]](https://huggingface.co/papers/2312.14238)

[\[📖 2.0 中文解读\]](https://zhuanlan.zhihu.com/p/706547971)  [\[📖 1.5 中文解读\]](https://zhuanlan.zhihu.com/p/699439759)  [\[📖 1.0 中文解读\]](https://zhuanlan.zhihu.com/p/702946079)

[Switch to the Chinese version (切换至中文版)](/README_zh.md)

<a href="https://trendshift.io/repositories/9803" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9803" alt="OpenGVLab%2FInternVL | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
<img height="55" alt="image" src="https://github.com/user-attachments/assets/bd62ab46-f0ea-40c6-ab10-7fde671716cc">

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/overall.png)

</div>

## News 🚀🚀🚀

- `2025/04/17`: We open-source the [data construction pipeline](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/tools/reasoning_data_pipeline) and [training scripts](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl3.0/mpo) of [MPO](https://huggingface.co/papers/2411.10442) and [VisualPRM](https://huggingface.co/papers/2503.10291). Additionally, the data construction scripts for [MPO](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl3.0/mpo_data_construction) and [VisualPRM](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl3.0/visualprm_data_construction) are also released for reference.
- `2025/04/11`: 🚀 We introduce [InternVL3](https://huggingface.co/collections/OpenGVLab/internvl3-67f7f690be79c2fe9d74fe9d), an advanced multimodal large language model (MLLM) series that demonstrates superior overall performance. InternVL3-78B achieves SoTA performance in both [perception](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME) and [reasoning performance](https://rank.opencompass.org.cn/leaderboard-multimodal-reasoning/?m=REALTIME) among open-source MLLMs. The key designs of InternVL3-78B include [Variable Visual Position Encoding](https://huggingface.co/papers/2412.09616), [Native Multimodal Pre-Training](https://huggingface.co/papers/2504.10479), [Mixed Preference Optimization](https://huggingface.co/papers/2411.10442), and [Multimodal Test-Time Scaling](https://huggingface.co/papers/2503.10291).
- `2025/03/13`: 🔥 We introduce [VisualPRM](https://huggingface.co/OpenGVLab/VisualPRM-8B), an advanced multimodal Process Reward Model (PRM) with 8B parameters, which improves the overall reasoning performance of InternVL2.5-8B and InternVL2.5-78B by 8.4 and 5.9 points, respectively. The training data for this model, termed [VisualPRM400K](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K), is also open-sourced. Please refer to our [paper](https://huggingface.co/papers/2503.10291) and [project page](https://internvl.github.io/blog/2025-03-13-VisualPRM/) for more details.
- `2024/12/20`: 🔥 We release the [InternVL2.5-MPO](https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/), which is finetuned with [Mixed Preference Optimization](https://huggingface.co/papers/2411.10442) on [MMPR-v1.1](https://huggingface.co/datasets/OpenGVLab/MMPR-v1.1). **The resulting models outperform their counterparts without MPO by an average of 2 points across all model scales on the OpenCompass leaderboard.** These models are available at [HF link](https://huggingface.co/collections/OpenGVLab/internvl25-mpo-6753fed98cd828219b12f849).
- `2024/12/17`: 🚀 [InternVL2/2.5](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/internvl2) is supported in [PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX) by Paddle Team.
- `2024/12/05`: 🚀 We release the [InternVL2.5](https://huggingface.co/collections/OpenGVLab/internvl-25-673e1019b66e2218f68d7c1c), an advanced multimodal large language model (MLLM) series with parameter coverage ranging from 1B to 78B. [InternVL2_5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B) is the first open-source MLLMs to achieve over **70%** on the **MMMU benchmark**, matching the performance of leading closed-source commercial models like GPT-4o. These models are available at [HF link](https://huggingface.co/collections/OpenGVLab/internvl-25-673e1019b66e2218f68d7c1c).
- `2024/11/14`: We introduce [MMPR](https://huggingface.co/datasets/OpenGVLab/MMPR), a high-quality, large-scale multimodal reasoning preference dataset, and [MPO](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl2.0_mpo), an effective preference optimization algorithm. The resulting model, [InternVL2-8B-MPO](https://huggingface.co/OpenGVLab/InternVL2-8B-MPO), achieves an accuracy of 67.0 on MathVista. Please refer to our [paper](https://arxiv.org/abs/2411.10442), [project page](https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/) and [document](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html) for more details.
- `2024/10/21`: We release the Mini-InternVL series. These models achieve impressive performance with minimal size: the 4B model achieves 90% of the performance with just 5% of the model size. For more details, please check our [project page](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/mini_internvl) and [document](https://internvl.readthedocs.io/en/latest/internvl2.0/domain_adaptation.html).
- `2024/08/01`: The [Chartmimic](https://chartmimic.github.io/) team evaluated the InternVL2 series models on their benchmark. The InternVL2-26B and 76B models achieved the top two performances among open-source models, with the InternVL2 76B model surpassing GeminiProVision and exhibiting comparable results to Claude-3-opus.
- `2024/08/01`: InternVL2-Pro achieved the SOTA performance among open-source models on the [CharXiv](https://charxiv.github.io/#leaderboard) dataset, surpassing many closed-source models such as GPT-4V, Gemini 1.5 Flash, and Claude 3 Sonnet.
- `2024/07/24`: The [MLVU](https://github.com/JUNJIE99/MLVU) team evaluated InternVL-1.5 on their benchmark. The average performance on the multiple-choice task was 50.4%, while the performance on the generative tasks was 4.02. The performance on the multiple-choice task ranked #1 among all open-source MLLMs.
- `2024/07/04`: We release the [InternVL2 series](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e). InternVL2-Pro achieved a 62.0% accuracy on the MMMU benchmark, matching the performance of leading closed-source commercial models like GPT-4o.

<details>
<summary>More News</summary>

- `2024/07/18`: InternVL2-40B achieved SOTA performance among open-source models on the [Video-MME](https://github.com/BradyFU/Video-MME) dataset, scoring 61.2 when inputting 16 frames and 64.4 when inputting 32 frames. It significantly outperforms other open-source models and is the closest open-source model to GPT-4o mini.
- `2024/07/18`: InternVL2-Pro achieved the SOTA performance on the [DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1) and [InfoVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=3) benchmarks.
- `2024/06/19`: We propose Needle In A Multimodal Haystack ([MM-NIAH](https://github.com/OpenGVLab/MM-NIAH)), the first benchmark designed to systematically evaluate the capability of existing MLLMs to comprehend long multimodal documents.
- `2024/05/30`: We release [ShareGPT-4o](https://sharegpt4o.github.io/), a large-scale dataset that we plan to open-source with 200K images, 10K videos, and 10K audios with detailed descriptions.
- `2024/05/28`: Thanks to the [lmdeploy](https://github.com/InternLM/lmdeploy) team for providing AWQ quantization support. The 4-bit model is available at [OpenGVLab/InternVL-Chat-V1-5-AWQ](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-AWQ).
- `2024/05/13`: InternVL 1.0 can now be used as the [text encoder](https://huggingface.co/OpenGVLab/InternVL-14B-224px) for diffusion models to support multilingual generation natively in over 110 languages worldwide. See [MuLan](https://github.com/mulanai/MuLan) for more details.
- `2024/04/18`: InternVL-Chat-V1-5 has been released at [HF link](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5), approaching the performance of GPT-4V and Gemini Pro on various benchmarks like MMMU, DocVQA, ChartQA, MathVista, etc.
- `2024/02/27`: InternVL is accepted by CVPR 2024 (Oral)! 🎉
- `2024/02/21`: [InternVL-Chat-V1-2-Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) achieved SOTA performance on MathVista (59.9), MMBench (83.8), and MMVP (58.7). See our [blog](https://internvl.github.io/blog/2024-02-21-InternVL-1.2/) for more details.
- `2024/02/12`: InternVL-Chat-V1-2 has been released. It achieves 51.6 on MMMU val and 82.3 on MMBench test. For more details, please refer to our [blog](https://internvl.github.io/blog/2024-02-21-InternVL-1.2/) and [SFT data](./internvl_chat#prepare-training-datasets). The model is now available on [HuggingFace](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2), and both training / evaluation data and scripts are open-sourced.
- `2024/01/24`: InternVL-Chat-V1-1 is released, it supports Chinese and has stronger OCR capability, see [here](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1).
- `2024/01/16`: We release our [customized mmcv/mmsegmentation/mmdetection code](https://github.com/OpenGVLab/InternVL-MMDetSeg), integrated with DeepSpeed, which can be used for training large-scale detection and segmentation models.

</details>

## Documents

### 🌟 **Get Started**

- **Installation**: 🌱 [Installation Guide](https://internvl.readthedocs.io/en/latest/get_started/installation.html) | 📄 [requirements.txt](./requirements.txt)
- **Chat Data Format**: 📝 [Meta File](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#meta-file) | ✏️ [Text](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#pure-text-data) | 🖼️ [Single-Image](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#single-image-data) | 🖼️🖼️ [Multi-Image](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#multi-image-data) | 🎥 [Video](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data)
- **Local Chat Demo**: 🤖 [Streamlit Demo](https://internvl.readthedocs.io/en/latest/get_started/local_chat_demo.html#streamlit-demo)
- **InternVL-Chat API**: 🌐 [InternVL2.5 API](https://internlm.intern-ai.org.cn/api/document)
- **Tutorials**: 🚀 [Enhancing InternVL2 on COCO Caption Using LoRA Fine-Tuning](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html)

### 🏆 **InternVL Family**

- **InternVL 3.0**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl3.0/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl3.0/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html) | 📊 [Evaluate](https://internvl.readthedocs.io/en/latest/internvl3.0/evaluation.html) | 📦 [Deploy](https://internvl.readthedocs.io/en/latest/internvl3.0/deployment.html) | 🎯 [MPO](https://internvl.readthedocs.io/en/latest/internvl3.0/preference_optimization.html)
- **InternVL 2.5**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl2.5/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl2.5/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html) | 📊 [Evaluate](https://internvl.readthedocs.io/en/latest/internvl2.5/evaluation.html) | 📦 [Deploy](https://internvl.readthedocs.io/en/latest/internvl2.5/deployment.html) | 🎯 [MPO](https://internvl.readthedocs.io/en/latest/internvl2.5/preference_optimization.html)
- **InternVL 2.0**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html) | 📊 [Evaluate](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html) | 📦 [Deploy](https://internvl.readthedocs.io/en/latest/internvl2.0/deployment.html) | 🎯 [MPO](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html)
- **InternVL 1.5**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl1.5/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl1.5/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl1.5/finetune.html) | 📊 [Evaluate](https://internvl.readthedocs.io/en/latest/internvl1.5/evaluation.html) | 📦 [Deploy](https://internvl.readthedocs.io/en/latest/internvl1.5/deployment.html)
- **InternVL 1.2**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl1.2/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl1.2/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl1.2/finetune.html) | 📊 [Evaluate](https://internvl.readthedocs.io/en/latest/internvl1.2/evaluation.html)
- **InternVL 1.1**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl1.1/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl1.1/quick_start.html) | 📊 [Evaluation](https://internvl.readthedocs.io/en/latest/internvl1.1/evaluation.html)
- **InternVL 1.0**: 🖼️ [Classification](https://internvl.readthedocs.io/en/latest/internvl1.0/classification.html) | 📊 [CLIP-Benchmark](https://internvl.readthedocs.io/en/latest/internvl1.0/clip_benchmark.html) | 🎨 [Segmentation](https://internvl.readthedocs.io/en/latest/internvl1.0/segmentation.html) | 💬 [Chat-LLaVA](https://internvl.readthedocs.io/en/latest/internvl1.0/internvl_chat_llava.html) | ✨ [InternVL-G](https://internvl.readthedocs.io/en/latest/internvl1.0/internvl_g.html)

## Model Zoo

#### Multimodal Large Language Model (InternVL 3.0)
<table>
  <tr>
    <th>Model Name</th>
    <th>Vision Part</th>
    <th>Language Part</th>
    <th>HF&nbsp;Link</th>
    <th>MS&nbsp;Link</th>
  </tr>
  <tr>
    <td>InternVL3-1B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT&#8209;300M&#8209;448px&#8209;V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B">Qwen2.5&#8209;0.5B</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL3-1B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL3-1B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL3-2B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B">Qwen2.5-1.5B</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL3-2B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL3-2B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL3-8B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-7B">Qwen2.5-7B</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL3-8B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL3-8B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL3-9B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/internlm/internlm3-8b-instruct">internlm3-8b-instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL3-9B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL3-9B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL3-14B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-14B">Qwen2.5-14B</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL3-14B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL3-14B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL3-38B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">InternViT-6B-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-32B">Qwen2.5-32B</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL3-38B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL3-38B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL3-78B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">InternViT-6B-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-72B">Qwen2.5-72B</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL3-78B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL3-78B">🤖 link</a></td>
  </tr>
</table>

#### Multimodal Large Language Model (InternVL 2.5)

<table>
  <tr>
    <th>Model Name</th>
    <th>Vision Part</th>
    <th>Language Part</th>
    <th>HF&nbsp;Link</th>
    <th>MS&nbsp;Link</th>
  </tr>
  <tr>
    <td>InternVL2_5-1B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT&#8209;300M&#8209;448px&#8209;V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct">Qwen2.5&#8209;0.5B&#8209;Instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-1B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-1B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-2B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/internlm/internlm2_5-1_8b-chat">internlm2_5-1_8b-chat</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-2B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-2B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-4B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct">Qwen2.5-3B-Instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-4B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-4B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-8B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/internlm/internlm2_5-7b-chat">internlm2_5-7b-chat</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-8B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-8B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-26B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">InternViT-6B-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/internlm/internlm2_5-20b-chat">internlm2_5-20b-chat</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-26B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-26B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-38B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">InternViT-6B-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct">Qwen2.5-32B-Instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-38B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-38B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-78B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">InternViT-6B-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct">Qwen2.5-72B-Instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-78B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-78B">🤖 link</a></td>
  </tr>
</table>

<table>
  <tr>
    <th>Model Name</th>
    <th>Vision Part</th>
    <th>Language Part</th>
    <th>HF&nbsp;Link</th>
    <th>MS&nbsp;Link</th>
  </tr>
  <tr>
    <td>InternVL2_5-1B-MPO</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT&#8209;300M&#8209;448px&#8209;V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct">Qwen2.5&#8209;0.5B&#8209;Instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-1B-MPO">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-1B-MPO">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-2B-MPO</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/internlm/internlm2_5-1_8b-chat">internlm2_5-1_8b-chat</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-2B-MPO">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-2B-MPO">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-4B-MPO</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct">Qwen2.5-3B-Instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-4B-MPO">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-4B-MPO">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-8B-MPO</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">InternViT-300M-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/internlm/internlm2_5-7b-chat">internlm2_5-7b-chat</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-8B-MPO">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-26B-MPO</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">InternViT-6B-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/internlm/internlm2_5-20b-chat">internlm2_5-20b-chat</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-26B-MPO">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-26B-MPO">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-38B-MPO</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">InternViT-6B-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct">Qwen2.5-32B-Instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-38B-MPO">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-38B-MPO">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2_5-78B-MPO</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">InternViT-6B-448px-V2_5</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct">Qwen2.5-72B-Instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-78B-MPO">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2_5-78B-MPO">🤖 link</a></td>
  </tr>
</table>

#### Multimodal Large Language Model (InternVL 2.0)

<table>
  <tr>
    <th>Model Name</th>
    <th>Vision Part</th>
    <th>Language Part</th>
    <th>HF&nbsp;Link</th>
    <th>MS&nbsp;Link</th>
  </tr>
  <tr>
    <td>InternVL2-1B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px">InternViT-300M-448px</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct">Qwen2-0.5B-Instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-1B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-1B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2-2B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px">InternViT-300M-448px</a></td>
    <td><a href="https://huggingface.co/internlm/internlm2-chat-1_8b">internlm2-chat-1-8b</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-2B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-2B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2-4B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px">InternViT-300M-448px</a></td>
    <td><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">Phi&#8209;3&#8209;mini&#8209;128k&#8209;instruct</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-4B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-4B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2-8B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px">InternViT-300M-448px</a></td>
    <td><a href="https://huggingface.co/internlm/internlm2_5-7b-chat">internlm2_5-7b-chat</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-8B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-8B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2-26B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5">InternViT-6B-448px-V1-5</a></td>
    <td><a href="https://huggingface.co/internlm/internlm2-chat-20b">internlm2-chat-20b</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-26B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-26B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2-40B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5">InternViT&#8209;6B&#8209;448px&#8209;V1&#8209;5</a></td>
    <td><a href="https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B">Nous&#8209;Hermes&#8209;2&#8209;Yi&#8209;34B</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-40B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-40B">🤖 link</a></td>
  </tr>
  <tr>
    <td>InternVL2&#8209;Llama3-76B</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5">InternViT-6B-448px-V1-5</a></td>
    <td><a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B">Hermes‑2‑Theta‑<br>Llama‑3‑70B</a></td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL2-Llama3-76B">🤖 link</a></td>
  </tr>
</table>

#### Multimodal Large Language Model (InternVL 1.0-1.5)

<table>
  <tr>
    <th>Model</th>
    <th>Date</th>
    <th>HF&nbsp;Link</th>
    <th>MS&nbsp;Link</th>
    <th>Note</th>
  </tr>
  <tr>
    <td>Mini&#8209;InternVL&#8209;Chat&#8209;4B&#8209;V1&#8209;5</td>
    <td>2024.05.28</td>
    <td><a href="https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/Mini-InternVL-Chat-4B-V1-5">🤖 link</a></td>
    <td>🚀🚀 16% of the model size, 90% of the performance</td>
  </tr>
  <tr>
    <td>Mini-InternVL-Chat-2B-V1-5</td>
    <td>2024.05.19</td>
    <td><a href="https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/Mini-InternVL-Chat-2B-V1-5">🤖 link</a></td>
    <td>🚀 8% of the model size, 80% of the performance</td>
  </tr>
  <tr>
    <td>InternVL-Chat-V1-5</td>
    <td>2024.04.18</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-V1-5">🤖 link</a></td>
    <td>support 4K image; super strong OCR; Approaching the performance of GPT-4V and Gemini Pro on various benchmarks like MMMU, DocVQA, ChartQA, MathVista, etc.</td>
  </tr>
  <tr>
    <td>InternVL-Chat-V1-2-Plus</td>
    <td>2024.02.21</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-V1-2-Plus">🤖 link</a></td>
    <td>more SFT data and stronger</td>
  </tr>
  <tr>
    <td>InternVL-Chat-V1-2</td>
    <td>2024.02.11</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-V1-2">🤖 link</a></td>
    <td>scaling up LLM to 34B</td>
  </tr>
  <tr>
    <td>InternVL-Chat-V1-1</td>
    <td>2024.01.24</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-V1-1">🤖 link</a></td>
    <td>support Chinese and stronger OCR</td>
  </tr>
  <tr>
    <td>InternVL-Chat-19B</td>
    <td>2023.12.25</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B">🤖 link</a></td>
    <td>English multimodal dialogue</td>
  </tr>
  <tr>
    <td>InternVL-Chat-13B</td>
    <td>2023.12.25</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B">🤖 link</a></td>
    <td>English multimodal dialogue</td>
  </tr>
</table>

#### CLIP-like Model (InternVL 1.0-2.5)

<table>
  <tr>
    <th>Model</th>
    <th>Date</th>
    <th>HF&nbsp;Link</th>
    <th>MS&nbsp;Link</th>
    <th>Note</th>
  </tr>
  <tr>
    <td>InternViT-300M-448px-V2_5</td>
    <td>2024.12.05</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-300M-448px-V2_5">🤖 link</a></td>
    <td>🚀🚀 A more powerful lightweight visual encoder. (🔥new)</td>
  </tr>
  <tr>
    <td>InternViT-6B-448px-V2_5</td>
    <td>2024.12.05</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-448px-V2_5">🤖 link</a></td>
    <td>🚀🚀 A stronger visual encoder to extract visual features. (🔥new)</td>
  </tr>
  <tr>
    <td>InternViT-300M-448px</td>
    <td>2024.05.25</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-300M-448px">🤖 link</a></td>
    <td>distilled small vision foundation model with 300M parameters </td>
  </tr>
  <tr>
    <td>InternViT&#8209;6B&#8209;448px&#8209;V1&#8209;5</td>
    <td>2024.04.20</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-448px-V1-5">🤖 link</a></td>
    <td>support dynamic resolution and super strong OCR feature extraction capability by incremental pre-training</td>
  </tr>
  <tr>
    <td>InternViT-6B-448px-V1-2</td>
    <td>2024.02.11</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-448px-V1-2">🤖 link</a></td>
    <td>support 448 resolution by incremental pre-training</td>
  </tr>
  <tr>
    <td>InternViT-6B-448px-V1-0</td>
    <td>2024.01.30</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-0">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-448px-V1-0">🤖 link</a></td>
    <td>support 448 resolution by incremental pre-training</td>
  </tr>
  <tr>
    <td>InternViT-6B-224px</td>
    <td>2023.12.22</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-224px">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-224px">🤖 link</a></td>
    <td>the first version of InternViT-6B, extracted from InternVL‑14B‑224px</td>
  </tr>
</table>

#### Vision-Language Foundation Model (InternVL 1.0)

<table>
  <tr>
    <th>Model</th>
    <th>Date</th>
    <th>HF&nbsp;Link</th>
    <th>MS&nbsp;Link</th>
    <th>Note</th>
  </tr>
  <tr>
    <td>InternVL&#8209;14B&#8209;224px</td>
    <td>2023.12.22</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-14B-224px">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-14B-224px">🤖 link</a></td>
    <td>vision-language foundation model, InternViT-6B + QLLaMA, can be used for image-text retrieval like CLIP</td>
  </tr>
</table>

## TODO List

- [x] Release training / evaluation code for InternVL2.5 series
- [x] Support liger kernels to save GPU memory
- [x] Release the code, model, and data of MPO
- [x] Support multimodal packed dataset
- [ ] Support vLLM and Ollama
- [ ] Support video and PDF input in online demo
- [ ] Release InternVL2 with VisionLLMv2 integration
- [x] Rebuild documents using readthedocs
- [x] Support fine-tuning different LLMs with LoRA
- [x] Release `requirements.txt` for InternVL2
- [x] Release training / evaluation code for InternVL2 series
- [x] Release Streamlit web UI for InternVL1.5 and InternVL2

## What can InternVL do?

<details>
  <summary>Visual Perception (click to expand)</summary>

- Linear-Probe Image Classification [\[see details\]](./classification#-evaluation)

  ViT-22B uses the private JFT-3B dataset.

  | method              | #param | IN-1K | IN-ReaL | IN-V2 | IN-A  | IN-R  | IN-Sketch |
  | ------------------- | :----: | :---: | :-----: | :---: | :---: | :---: | :-------: |
  | OpenCLIP-G          |  1.8B  | 86.2  |  89.4   | 77.2  | 63.8  | 87.8  |   66.4    |
  | DINOv2-g            |  1.1B  | 86.5  |  89.6   | 78.4  | 75.9  | 78.8  |   62.5    |
  | EVA-01-CLIP-g       |  1.1B  | 86.5  |  89.3   | 77.4  | 70.5  | 87.7  |   63.1    |
  | MAWS-ViT-6.5B       |  6.5B  | 87.8  |    -    |   -   |   -   |   -   |     -     |
  | ViT-22B\*           | 21.7B  | 89.5  |  90.9   | 83.2  | 83.8  | 87.4  |     -     |
  | InternViT-6B (ours) |  5.9B  | 88.2  |  90.4   | 79.9  | 77.5  | 89.8  |   69.1    |

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

  | method            | IN-1K | IN-A  | IN-R  | IN-V2 | IN-Sketch | ObjectNet |
  | ----------------- | :---: | :---: | :---: | :---: | :-------: | :-------: |
  | OpenCLIP-G        | 80.1  | 69.3  | 92.1  | 73.6  |   68.9    |   73.0    |
  | EVA-02-CLIP-E+    | 82.0  | 82.1  | 94.5  | 75.7  |   71.6    |   79.6    |
  | ViT-22B\*         | 85.9  | 90.1  | 96.0  | 80.9  |     -     |   87.6    |
  | InternVL-C (ours) | 83.2  | 83.8  | 95.5  | 77.3  |   73.9    |   80.6    |

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

- Zero-Shot Video Classification

  | method            | #frame | K400  | K600  | K700  |
  | ----------------- | :----: | :---: | :---: | :---: |
  | OpenCLIP-G        |   1    | 65.9  | 66.1  | 59.2  |
  | EVA-02-CLIP-E+    |   1    | 69.8  | 69.3  | 63.4  |
  | InternVL-C (ours) |   1    | 71.0  | 71.3  | 65.7  |
  | ViCLIP            |   8    | 75.7  | 73.5  | 66.4  |
  | InternVL-C (ours) |   8    | 79.4  | 78.8  | 71.5  |

</details>

<details>
  <summary>Cross-Modal Retrieval (click to expand)</summary>

- English Zero-Shot Image-Text Retrieval [\[see details\]](./clip_benchmark#flickr30k--coco)

  <table>
    <tr align=center>
        <td rowspan="3" align=left><b>model</b></td>
        <td colspan="6" align=center><b>Flickr30K</b></td>
        <td colspan="6" align=center><b>COCO</b></td>
        <td rowspan="3" align=center><b>avg</b></td>
    </tr>
     <tr align=center>
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

  | method            |  EN   |  ES   |  FR   |  ZH   |  IT   |  KO   |  RU   |  JP   | average |
  | ----------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-----: |
  | AltCLIP           | 95.4  | 94.1  | 92.9  | 95.1  | 94.2  | 94.4  | 91.8  | 91.7  |  93.7   |
  | OpenCLIP-XLM-R-H  | 97.3  | 96.1  | 94.5  | 94.7  | 96.0  | 90.2  | 93.9  | 94.0  |  94.6   |
  | InternVL-C (ours) | 97.3  | 95.7  | 95.1  | 95.6  | 96.0  | 92.2  | 93.3  | 95.5  |  95.1   |
  | InternVL-G (ours) | 98.6  | 97.7  | 96.5  | 96.7  | 96.9  | 95.1  | 94.8  | 96.1  |  96.6   |

</details>

<details>
  <summary>Multimodal Dialogue</summary>

</details>

## Quick Start with HuggingFace

<details>
  <summary>using InternViT-6B for visual feature extraction (click to expand)</summary>

```python
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-6B-448px-V2_5',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda().eval()

image = Image.open('./examples/image1.jpg').convert('RGB')

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-6B-448px-V1-5')

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

outputs = model(pixel_values)
```

</details>

<details>
  <summary>using InternVL-C(ontrastive) and InternVL-G(enerative) for cross-modal retrieval (click to expand)</summary>

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
  <summary>using InternVL 2.5 for multimodal chat (click to expand)</summary>

Here, we take the smaller `OpenGVLab/InternVL2_5-8B` as an example:

```python
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'OpenGVLab/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=False)

# pure-text conversation (纯文本对话)
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Can you tell me a story?'
response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# single-image single-round conversation (单图单轮对话)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')

# single-image multi-round conversation (单图多轮对话)
question = '<image>\nPlease describe the image in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Please write a poem according to the image.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

question = '<image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# batch inference, single image per sample (单图批处理)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
responses = model.batch_chat(tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}\nAssistant: {response}')

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

video_path = './examples/red-panda.mp4'
pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
video_prefix = ''.join([f'Frame-{i+1}: <image>\n' for i in range(len(num_patches_list))])
question = video_prefix + 'What is the red panda doing?'
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Describe this video in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
```

</details>

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{zhu2025internvl3exploringadvancedtraining,
      title={InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models}, 
      author={Jinguo Zhu and Weiyun Wang and Zhe Chen and Zhaoyang Liu and Shenglong Ye and Lixin Gu and Hao Tian and Yuchen Duan and Weijie Su and Jie Shao and Zhangwei Gao and Erfei Cui and Xuehui Wang and Yue Cao and Yangzhou Liu and Xingguang Wei and Hongjie Zhang and Haomin Wang and Weiye Xu and Hao Li and Jiahao Wang and Nianchen Deng and Songze Li and Yinan He and Tan Jiang and Jiapeng Luo and Yi Wang and Conghui He and Botian Shi and Xingcheng Zhang and Wenqi Shao and Junjun He and Yingtong Xiong and Wenwen Qu and Peng Sun and Penglong Jiao and Han Lv and Lijun Wu and Kaipeng Zhang and Huipeng Deng and Jiaye Ge and Kai Chen and Limin Wang and Min Dou and Lewei Lu and Xizhou Zhu and Tong Lu and Dahua Lin and Yu Qiao and Jifeng Dai and Wenhai Wang},
      year={2025},
      eprint={2504.10479},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.10479}, 
}
@article{chen2024expanding,
  title={Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling},
  author={Chen, Zhe and Wang, Weiyun and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Cui, Erfei and Zhu, Jinguo and Ye, Shenglong and Tian, Hao and Liu, Zhaoyang and others},
  journal={arXiv preprint arXiv:2412.05271},
  year={2024}
}
@article{wang2024mpo,
  title={Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization},
  author={Wang, Weiyun and Chen, Zhe and Wang, Wenhai and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Zhu, Jinguo and Zhu, Xizhou and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2411.10442},
  year={2024}
}
@article{gao2024mini,
  title={Mini-InternVL: a flexible-transfer pocket multi-modal model with 5\% parameters and 90\% performance},
  author={Gao, Zhangwei and Chen, Zhe and Cui, Erfei and Ren, Yiming and Wang, Weiyun and Zhu, Jinguo and Tian, Hao and Ye, Shenglong and He, Junjun and Zhu, Xizhou and others},
  journal={Visual Intelligence},
  volume={2},
  number={1},
  pages={1--17},
  year={2024},
  publisher={Springer}
}
@article{chen2024far,
  title={How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={Science China Information Sciences},
  volume={67},
  number={12},
  pages={220101},
  year={2024},
  publisher={Springer}
}
@inproceedings{chen2024internvl,
  title={Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24185--24198},
  year={2024}
}
```

## Acknowledgement

InternVL is built with reference to the code of the following projects: [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip), [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), [EVA](https://github.com/baaivision/EVA/tree/master), [InternImage](https://github.com/OpenGVLab/InternImage), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Transformers](https://github.com/huggingface/transformers), [DINOv2](https://github.com/facebookresearch/dinov2), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm), and [LLaVA-1.5](https://github.com/haotian-liu/LLaVA). Thanks for their awesome work!

______________________________________________________________________

Scan the following QR Code, join our WeChat group.

<p align="center"><img width="300" alt="image" src="https://github.com/user-attachments/assets/f776df09-ebba-4fd5-80c2-fec4ff1518be"></p>
