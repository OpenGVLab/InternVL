<div align="center">

# InternVL家族：通过开源组件缩小与商业多模态模型的差距 —— GPT-4o的开源替代方案

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/930e6814-8a9f-43e1-a284-118a5732daa4">
  <br>
</div>

[\[🆕 博客\]](https://internvl.github.io/blog/) [\[🤔 常见问题\]](https://internvl.readthedocs.io/en/latest/tutorials/faqs.html)  [\[🗨️ 对话Demo\]](https://internvl.opengvlab.com/)  [\[🤗 HF Demo\]](https://huggingface.co/spaces/OpenGVLab/InternVL)  [\[📖 文档\]](https://internvl.readthedocs.io/en/latest/)  [\[🌐 API\]](https://internlm.intern-ai.org.cn/api/document)  [\[🚀 快速开始\]](#使用-huggingface-快速开始)

[\[📜 InternVL 2.5 报告\]](https://huggingface.co/papers/2412.05271) [\[🔥 Mini-InternVL 论文\]](https://arxiv.org/abs/2410.16261)  [\[🚀 InternVL2 博客\]](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)   [\[📜 InternVL 1.5 论文\]](https://huggingface.co/papers/2404.16821)  [\[📜 InternVL 1.0 论文\]](https://huggingface.co/papers/2312.14238)

[\[📖 2.0 中文解读\]](https://zhuanlan.zhihu.com/p/706547971)  [\[📖 1.5 中文解读\]](https://zhuanlan.zhihu.com/p/699439759)  [\[📖 1.0 中文解读\]](https://zhuanlan.zhihu.com/p/702946079)

[Switch to the English version (切换至英文版)](/README.md)

<a href="https://trendshift.io/repositories/9803" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9803" alt="OpenGVLab%2FInternVL | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
<img height="55" alt="image" src="https://github.com/user-attachments/assets/bd62ab46-f0ea-40c6-ab10-7fde671716cc">

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/overall.png)

</div>

## 最新消息 🚀🚀🚀

- `2025/04/17`: 我们开源了 [MPO](https://huggingface.co/papers/2411.10442) 和 [VisualPRM](https://huggingface.co/papers/2503.10291) 的[数据构造管线](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/tools/reasoning_data_pipeline)及[训练脚本](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl3.0/mpo)。 此外 [MPO](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl3.0/mpo_data_construction) 和 [VisualPRM](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl3.0/visualprm_data_construction) 的数据构建脚本也已经开源。
- `2025/04/11`: 🚀 我们发布了 [InternVL3](https://huggingface.co/collections/OpenGVLab/internvl3-67f7f690be79c2fe9d74fe9d)， 一个性能强大的开源多模态大模型。 其中 InternVL3-78B 同时在[感知能力](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME)和[推理能力](https://rank.opencompass.org.cn/leaderboard-multimodal-reasoning/?m=REALTIME)上同时达到了开源第一的性能。 InternVL3-78B 的核心技术包括：[Variable Visual Position Encoding](https://huggingface.co/papers/2412.09616)，[Native Multimodal Pre-Training](https://huggingface.co/papers/2504.10479)，[Mixed Preference Optimization](https://huggingface.co/papers/2411.10442)，以及 [Multimodal Test-Time Scaling](https://huggingface.co/papers/2503.10291)。
- `2025/03/13`: 🔥 我们发布了 [VisualPRM](https://huggingface.co/OpenGVLab/VisualPRM-8B)，一个8B参数两的多模态过程奖励模型（PRM）。该模型在 Best-of-8 的评测设置下使得 InternVL2.5-8B 和 InternVL2.5-78B 在七个多模态推理评测基准上的综合性能分别提升了 8.4 和 5.9 分。该模型的训练数据 [VisualPRM400K](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K)也已经开源。请参考我们的[论文](https://huggingface.co/papers/2503.10291)和[项目主页](https://internvl.github.io/blog/2025-03-13-VisualPRM/)来了解更多细节。
- `2024/12/20`: 🔥 我们发布了 [InternVL2.5-MPO系列](https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/)。该系列通过 [Mixed Preference Optimization](https://huggingface.co/papers/2411.10442) 算法和 [MMPR-v1.1](https://huggingface.co/datasets/OpenGVLab/MMPR-v1.1) 数据集微调得到。**该系列模型在OpenCompass评测榜单中的整体性能超过MPO训练前两个百分点。** 这些模型可在 [HF 链接](https://huggingface.co/collections/OpenGVLab/internvl25-mpo-6753fed98cd828219b12f849)中下载。
- `2024/12/17`: 🚀 Paddle团队已在[PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX)框架中适配[InternVL2/2.5](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/internvl2)。
- `2024/12/05`: 🚀 我们发布了 InternVL2.5 系列，覆盖了从1B参数到78B参数的多模态大语言模型。[InternVL2_5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B) 是首个在MMMU benchmark上得分超过70的开源模型。 这些模型可在 [HF 链接](https://huggingface.co/collections/OpenGVLab/internvl-25-673e1019b66e2218f68d7c1c) 中下载。
- `2024/11/14`: 我们发布了 [MMPR](https://huggingface.co/datasets/OpenGVLab/MMPR)，一个高质量、大规模的多模态推理偏好数据集，以及 [MPO](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/internvl2.0_mpo)，一种高效的偏好优化算法。由此训练的模型 [InternVL2-8B-MPO](https://huggingface.co/OpenGVLab/InternVL2-8B-MPO) 在 MathVista 上取得了 67.0 的准确率。更多详情请参阅我们的[论文](https://arxiv.org/abs/2411.10442)、[项目主页](https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/) 和 [文档](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html)。
- `2024/10/21`: 我们发布了 Mini-InternVL 系列。这些模型在保持极小模型体积的同时实现了出色的性能：4B 模型仅用 5% 的模型大小便达到了 90% 的性能。有关更多详细信息，请查看我们的 [项目页面](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat/shell/mini_internvl) 和 [文档](https://internvl.readthedocs.io/en/latest/internvl2.0/domain_adaptation.html)。
- `2024/08/01`: [Chartmimic](https://chartmimic.github.io/) 团队在他们的基准测试中评估了 InternVL2 系列模型。InternVL2-26B 和 76B 模型在开源模型中取得了前两名的成绩，其中 InternVL2-Llama3-76B 模型超过了 GeminiProVision，并表现出与 Claude-3-opus 相当的结果。
- `2024/08/01`: InternVL2-Pro 在 [CharXiv](https://charxiv.github.io/#leaderboard) 数据集中实现了开源模型中的 SOTA 性能，也比部分知名闭源模型如 GPT-4V、Gemini 1.5 Flash、Claude 3 Sonnet 取得了更好成绩
- `2024/07/24`: [MLVU](https://github.com/JUNJIE99/MLVU)团队在它们的基准测试中评估了InternVL-1.5。在多项选择任务上的平均表现为50.4%，而在生成任务上的表现为4.02。多项选择任务的表现在所有开源多模态大语言模型中排名第一。
- `2024/07/04`: 我们发布了 InternVL2 系列模型。InternVL2-Pro 在 MMMU 基准测试中达到了 62.0% 的准确率，实现了与 GPT-4o 等领先闭源商业模型比肩的性能。模型权重可在 [HF 链接](https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e) 中下载。

<details>
<summary>更多</summary>

- `2024/07/18`: InternVL2-40B 在 [Video-MME](https://github.com/BradyFU/Video-MME) 数据集中实现了开源模型中的 SOTA 性能，当输入 16 帧时得分为 61.2，输入 32 帧时得分为 64.4，大幅领先其它开源模型，是最接近 GPT-4o mini 的开源模型。
- `2024/07/18`: InternVL2-Pro 在 [DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1) 和 [InfoVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=3) 的基准测试中实现了 SOTA 性能。
- `2024/06/19`: 我们提出了 Needle In A Multimodal Haystack ([MM-NIAH](https://github.com/OpenGVLab/MM-NIAH))，这是第一个针对模型关于长多模态文档理解能力的评测基准。
- `2024/05/30`: 我们发布了 [ShareGPT-4o](https://sharegpt4o.github.io/)，这是一个大规模、高质量的多模态数据集。我们计划开源一批使用 GPT-4o 精心标注的数据，包括 200K 条图像详细描述、10K 条视频详细描述，以及 10K 条音频详细描述。
- `2024/05/29`: 我们开源了 Mini-InternVL 系列，包括以下两个对话模型：[Mini-InternVL-Chat-2B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5) 和 [Mini-InternVL-Chat-4B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5)。这些模型在极小的尺寸下实现了令人印象深刻的性能：2B 模型以 8% 的模型尺寸实现了 80% 的性能，4B 模型以 16% 的模型尺寸实现了 90% 的性能。更多细节请查看我们的[博客](https://internvl.github.io/blog/2024-05-25-Mini-InternVL-1.5/)。
- `2024/05/13`: InternVL 1.0 现在可以作为扩散模型的 [文本编码器](https://huggingface.co/OpenGVLab/InternVL-14B-224px)，支持全球超过 110 种语言的多语言生成。详情请看 [MuLan](https://github.com/mulanai/MuLan)。
- `2024/04/18`: InternVL-Chat-V1-5 已经在 [HuggingFace](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) 发布，在 MMMU、DocVQA、ChartQA、MathVista 等各种基准测试中，性能接近 GPT-4V 和 Gemini Pro。
- `2024/02/27`: InternVL 已被 CVPR 2024 (Oral) 接收！🎉
- `2024/02/21`: [InternVL-Chat-V1-2-Plus](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus) 在 MathVista（59.9）、MMBench（83.8）和 MMVP（58.7）上实现了 SOTA 性能。详情请看我们的[博客](https://internvl.github.io/blog/2024-02-21-InternVL-1.2/)。
- `2024/02/12`: InternVL-Chat-V1-2 已经发布，它在 MMMU 验证集上达到了 51.6，在 MMBench 测试集上达到了 82.3。 更多信息请参考我们的[博客](https://internvl.github.io/blog/2024-02-21-InternVL-1.2/)以及 [SFT 数据](./internvl_chat#prepare-training-datasets)。该模型已经在 [HuggingFace](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2) 发布，训练、测评的数据和脚本均已开源。
- `2024/01/24`: InternVL-Chat-V1-1 已经发布，它支持中文对话，并具备强大的 OCR 能力，详情请看[这里](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1)。
- `2024/01/16`: 我们发布了 [定制的 mmcv/mmsegmentation/mmdetection 代码库](https://github.com/OpenGVLab/InternVL-MMDetSeg)，集成了 DeepSpeed，可以用于训练检测和分割大模型。

</details>

## 使用文档

### 🌟 **Get Started**

- **Installation**: 🌱 [Installation Guide](https://internvl.readthedocs.io/en/latest/get_started/installation.html) | 📄 [requirements.txt](./requirements.txt)
- **Chat Data Format**: 📝 [Meta File](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#meta-file) | ✏️ [Text](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#pure-text-data) | 🖼️ [Single-Image](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#single-image-data) | 🖼️🖼️ [Multi-Image](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#multi-image-data) | 🎥 [Video](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html#video-data)
- **Local Chat Demo**: 🤖 [Streamlit Demo](https://internvl.readthedocs.io/en/latest/get_started/local_chat_demo.html#streamlit-demo)
- **InternVL-Chat API**: 🌐 [InternVL2.5 API](https://internlm.intern-ai.org.cn/api/document)
- **Tutorials**: 🚀 [Enhancing InternVL2 on COCO Caption Using LoRA Fine-Tuning](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html)

### 🏆 **InternVL Family**

- **InternVL 2.5**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl2.5/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl2.5/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html) | 📊 [Evaluate](https://internvl.readthedocs.io/en/latest/internvl2.5/evaluation.html) | 📦 [Deploy](https://internvl.readthedocs.io/en/latest/internvl2.5/deployment.html) | 🎯 [MPO](https://internvl.readthedocs.io/en/latest/internvl2.5/preference_optimization.html)
- **InternVL 2.0**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html) | 📊 [Evaluate](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html) | 📦 [Deploy](https://internvl.readthedocs.io/en/latest/internvl2.0/deployment.html) | 🎯 [MPO](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html)
- **InternVL 1.5**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl1.5/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl1.5/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl1.5/finetune.html) | 📊 [Evaluate](https://internvl.readthedocs.io/en/latest/internvl1.5/evaluation.html) | 📦 [Deploy](https://internvl.readthedocs.io/en/latest/internvl1.5/deployment.html)
- **InternVL 1.2**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl1.2/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl1.2/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl1.2/finetune.html) | 📊 [Evaluate](https://internvl.readthedocs.io/en/latest/internvl1.2/evaluation.html)
- **InternVL 1.1**: 📖 [Intro](https://internvl.readthedocs.io/en/latest/internvl1.1/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl1.1/quick_start.html) | 📊 [Evaluation](https://internvl.readthedocs.io/en/latest/internvl1.1/evaluation.html)
- **InternVL 1.0**: 🖼️ [Classification](https://internvl.readthedocs.io/en/latest/internvl1.0/classification.html) | 📊 [CLIP-Benchmark](https://internvl.readthedocs.io/en/latest/internvl1.0/clip_benchmark.html) | 🎨 [Segmentation](https://internvl.readthedocs.io/en/latest/internvl1.0/segmentation.html) | 💬 [Chat-LLaVA](https://internvl.readthedocs.io/en/latest/internvl1.0/internvl_chat_llava.html) | ✨ [InternVL-G](https://internvl.readthedocs.io/en/latest/internvl1.0/internvl_g.html)

## 模型库

#### 多模态大语言模型 (InternVL 2.5)

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

#### 多模态大语言模型 (InternVL 2.0)

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

#### 多模态大语言模型 (InternVL 1.0-1.5)

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
    <td>🚀🚀 16% 的模型大小, 90% 的性能</td>
  </tr>
  <tr>
    <td>Mini-InternVL-Chat-2B-V1-5</td>
    <td>2024.05.19</td>
    <td><a href="https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/Mini-InternVL-Chat-2B-V1-5">🤖 link</a></td>
    <td>🚀 8% 的模型大小, 80% 的性能</td>
  </tr>
  <tr>
    <td>InternVL-Chat-V1-5</td>
    <td>2024.04.18</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-V1-5">🤖 link</a></td>
    <td>支持 4K 图像；超强的 OCR 能力；在 MMMU、DocVQA、ChartQA、MathVista 等各种基准测试中，性能接近 GPT-4V 和 Gemini Pro
  </tr>
  <tr>
    <td>InternVL-Chat-V1-2-Plus</td>
    <td>2024.02.21</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-V1-2-Plus">🤖 link</a></td>
    <td>更多的 SFT 数据和更强的性能</td>
  </tr>
  <tr>
    <td>InternVL-Chat-V1-2</td>
    <td>2024.02.11</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-V1-2">🤖 link</a></td>
    <td>将 LLM 扩展到 34B</td>
  </tr>
  <tr>
    <td>InternVL-Chat-V1-1</td>
    <td>2024.01.24</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-1">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-V1-1">🤖 link</a></td>
    <td>支持中文和更强的 OCR 能力</td>
  </tr>
  <tr>
    <td>InternVL-Chat-19B</td>
    <td>2023.12.25</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-13B">🤖 link</a></td>
    <td>英语多模态对话</td>
  </tr>
  <tr>
    <td>InternVL-Chat-13B</td>
    <td>2023.12.25</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B">🤖 link</a></td>
    <td>英语多模态对话</td>
  </tr>
</table>

#### 类 CLIP 模型 (InternVL 1.0-2.5)

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
    <td>🚀🚀 一个更强的轻量视觉编码器 (🔥新)</td>
  </tr>
  <tr>
    <td>InternViT-6B-448px-V2_5</td>
    <td>2024.12.05</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-448px-V2_5">🤖 link</a></td>
    <td>🚀🚀 拥有更强的视觉特征提取能力 (🔥新)</td>
  </tr>
  <tr>
    <td>InternViT-300M-448px</td>
    <td>2024.05.25</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-300M-448px">🤖 link</a></td>
    <td>蒸馏的小型视觉基础模型，具有 300M 参数</td>
  </tr>
  <tr>
    <td>InternViT&#8209;6B&#8209;448px&#8209;V1&#8209;5</td>
    <td>2024.04.20</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-448px-V1-5">🤖 link</a></td>
    <td>通过增量预训练支持动态分辨率和超强的 OCR 特征提取能力</td>
  </tr>
  <tr>
    <td>InternViT-6B-448px-V1-2</td>
    <td>2024.02.11</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-448px-V1-2">🤖 link</a></td>
    <td>通过增量预训练支持 448 分辨率</td>
  </tr>
  <tr>
    <td>InternViT-6B-448px-V1-0</td>
    <td>2024.01.30</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-0">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-448px-V1-0">🤖 link</a></td>
    <td>通过增量预训练支持 448 分辨率</td>
  </tr>
  <tr>
    <td>InternViT-6B-224px</td>
    <td>2023.12.22</td>
    <td><a href="https://huggingface.co/OpenGVLab/InternViT-6B-224px">🤗 link</a></td>
    <td><a href="https://modelscope.cn/models/OpenGVLab/InternViT-6B-224px">🤖 link</a></td>
    <td>InternViT-6B 的第一个版本，提取自 InternVL‑14B‑224px</td>
  </tr>
</table>

#### 视觉语言基础模型 (InternVL 1.0)

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
    <td>视觉-语言基础模型，InternViT-6B + QLLaMA，可以用于类似 CLIP 的图文检索</td>
  </tr>
</table>

## TODO 列表

- [x] 发布 InternVL2.5 系列的训练 / 评估代码
- [x] 支持 liger kernels 以节省显存
- [x] 发布 MPO 的代码、模型和数据
- [x] 支持多模态 packed dataset
- [ ] 支持 vLLM 和 Ollama
- [ ] 在 Demo 中支持视频和 PDF 输入
- [ ] 发布集成 VisionLLMv2 的 InternVL2
- [x] 使用 readthedocs 重新构建文档
- [x] 支持使用 LoRA 微调不同的 LLMs
- [x] 发布 InternVL2 的 `requirements.txt`
- [x] 发布 InternVL2 系列的训练 / 评估代码
- [x] 发布 InternVL1.5 和 InternVL2 的 Streamlit 网页 UI

## InternVL 可以做什么?

<details>
  <summary>视觉感知 (点击展开)</summary>

- 线性探针图像分类 [\[查看详情\]](./classification#-evaluation)

  ViT-22B uses the private JFT-3B dataset.

  | method              | #param | IN-1K | IN-ReaL | IN-V2 | IN-A  | IN-R  | IN-Sketch |
  | ------------------- | :----: | :---: | :-----: | :---: | :---: | :---: | :-------: |
  | OpenCLIP-G          |  1.8B  | 86.2  |  89.4   | 77.2  | 63.8  | 87.8  |   66.4    |
  | DINOv2-g            |  1.1B  | 86.5  |  89.6   | 78.4  | 75.9  | 78.8  |   62.5    |
  | EVA-01-CLIP-g       |  1.1B  | 86.5  |  89.3   | 77.4  | 70.5  | 87.7  |   63.1    |
  | MAWS-ViT-6.5B       |  6.5B  | 87.8  |    -    |   -   |   -   |   -   |     -     |
  | ViT-22B\*           | 21.7B  | 89.5  |  90.9   | 83.2  | 83.8  | 87.4  |     -     |
  | InternViT-6B (ours) |  5.9B  | 88.2  |  90.4   | 79.9  | 77.5  | 89.8  |   69.1    |

- 语义分割 [\[查看详情\]](./segmentation#-evaluation)

  | method                | decoder | #param (train/total) | crop size | mIoU         |
  | --------------------- | :-----: | :------------------: | :-------: | ------------ |
  | OpenCLIP-G (frozen)   | Linear  |     0.3M / 1.8B      |    512    | 39.3         |
  | ViT-22B (frozen)      | Linear  |     0.9M / 21.7B     |    504    | 34.6         |
  | InternViT-6B (frozen) | Linear  |     0.5M / 5.9B      |    504    | 47.2 (+12.6) |
  | ViT-22B (frozen)      | UperNet |     0.8B / 22.5B     |    504    | 52.7         |
  | InternViT-6B (frozen) | UperNet |     0.4B / 6.3B      |    504    | 54.9 (+2.2)  |
  | ViT-22B               | UperNet |    22.5B / 22.5B     |    504    | 55.3         |
  | InternViT-6B          | UperNet |     6.3B / 6.3B      |    504    | 58.9 (+3.6)  |

- 零样本图像分类 [\[查看详情\]](./clip_benchmark#imagenet-variants-and-objectnet)

  | method            | IN-1K | IN-A  | IN-R  | IN-V2 | IN-Sketch | ObjectNet |
  | ----------------- | :---: | :---: | :---: | :---: | :-------: | :-------: |
  | OpenCLIP-G        | 80.1  | 69.3  | 92.1  | 73.6  |   68.9    |   73.0    |
  | EVA-02-CLIP-E+    | 82.0  | 82.1  | 94.5  | 75.7  |   71.6    |   79.6    |
  | ViT-22B\*         | 85.9  | 90.1  | 96.0  | 80.9  |     -     |   87.6    |
  | InternVL-C (ours) | 83.2  | 83.8  | 95.5  | 77.3  |   73.9    |   80.6    |

- 多语言零样本图像分类 [\[查看详情\]](./clip_benchmark#multilingual-imagenet-1k)

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

- 零样本视频分类

  | method            | #frame | K400  | K600  | K700  |
  | ----------------- | :----: | :---: | :---: | :---: |
  | OpenCLIP-G        |   1    | 65.9  | 66.1  | 59.2  |
  | EVA-02-CLIP-E+    |   1    | 69.8  | 69.3  | 63.4  |
  | InternVL-C (ours) |   1    | 71.0  | 71.3  | 65.7  |
  | ViCLIP            |   8    | 75.7  | 73.5  | 66.4  |
  | InternVL-C (ours) |   8    | 79.4  | 78.8  | 71.5  |

</details>

<details>
  <summary>跨模态检索 (点击展开)</summary>

- 英语零样本图文检索 [\[查看详情\]](./clip_benchmark#flickr30k--coco)

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

- 中文零样本图文检索 [\[查看详情\]](./clip_benchmark#flickr30k-cn--coco-cn)

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

- 多语言零样本图文对检索 [\[查看详情\]](./clip_benchmark#xtd)

  | method            |  EN   |  ES   |  FR   |  ZH   |  IT   |  KO   |  RU   |  JP   | average |
  | ----------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :-----: |
  | AltCLIP           | 95.4  | 94.1  | 92.9  | 95.1  | 94.2  | 94.4  | 91.8  | 91.7  |  93.7   |
  | OpenCLIP-XLM-R-H  | 97.3  | 96.1  | 94.5  | 94.7  | 96.0  | 90.2  | 93.9  | 94.0  |  94.6   |
  | InternVL-C (ours) | 97.3  | 95.7  | 95.1  | 95.6  | 96.0  | 92.2  | 93.3  | 95.5  |  95.1   |
  | InternVL-G (ours) | 98.6  | 97.7  | 96.5  | 96.7  | 96.9  | 95.1  | 94.8  | 96.1  |  96.6   |

</details>

<details>
  <summary>多模态对话</summary>

</details>

## 使用 HuggingFace 快速开始

<details>
  <summary>使用 InternViT-6B 提取视觉特征 (点击展开)</summary>

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
  <summary>使用 InternVL-C(ontrastive) 和 InternVL-G(enerative) 进行跨模态检索 (点击展开)</summary>

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
  <summary>使用 InternVL 2.5 进行多模态对话 (点击展开)</summary>

这里我们以较小的 `OpenGVLab/InternVL2_5-8B` 为例：

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

## 许可证

本项目以 [MIT 许可证](LICENSE) 发布。项目中的部分代码和模型来自其它来源，受其原始许可证的约束。

## 引用

如果您在研究中发现本项目有用，请考虑引用：

```BibTeX
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

## 致谢

InternVL 的代码构建参考了以下的项目: [OpenAI CLIP](https://github.com/openai/CLIP)、[Open CLIP](https://github.com/mlfoundations/open_clip)、[CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)、[EVA](https://github.com/baaivision/EVA/tree/master)、[InternImage](https://github.com/OpenGVLab/InternImage)、[ViT-Adapter](https://github.com/czczup/ViT-Adapter)、[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)、[Transformers](https://github.com/huggingface/transformers)、[DINOv2](https://github.com/facebookresearch/dinov2)、[BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)、[Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm)和 [LLaVA-1.5](https://github.com/haotian-liu/LLaVA)，感谢这些杰出的工作。

______________________________________________________________________

扫描下方二维码，加入我们的项目微信群。

<p align="center"><img width="300" alt="image" src="https://github.com/user-attachments/assets/f776df09-ebba-4fd5-80c2-fec4ff1518be"></p>
