# InternVL-Chat

This folder contains the implementation of the training code for [InternVL3_5-GPT-OSS-20B-A4B](https://huggingface.co/OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview).
We provide a [conda environment package](https://huggingface.co/Weiyun1025/InternVL3_5-GPT-OSS-conda-env) along with corresponding [training scripts](shell/internvl3_5_gpt_oss).
The efficient implementation of GptOssAttention is provided in this [folder](internvl/patch/flash_sink_attn), with credit to [Wenhao Li](https://github.com/wenhaoli-xmu).
Please refer to the model card for more details.

## 📖 Documents

### 🌟 **Get Started**

- **Installation**: 🌱 [conda environment archive](https://huggingface.co/Weiyun1025/InternVL3_5-GPT-OSS-conda-env) | 📄 [requirements.txt](./requirements.txt)
- **Tutorials**: 🚀 [Enhancing InternVL on COCO Caption Using LoRA Fine-Tuning](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html)

### 🏆 **InternVL Family**

- **InternVL 3.5**: 📖 [Introduction](https://internvl.readthedocs.io/en/latest/internvl3.0/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl3.0/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html) | 📊 [Evaluation](https://internvl.readthedocs.io/en/latest/internvl3.0/evaluation.html) | 📦 [Deployment](https://internvl.readthedocs.io/en/latest/internvl3.0/deployment.html) | 🎯 [Preference Optimization](https://internvl.readthedocs.io/en/latest/internvl3.0/preference_optimization.html)
- **InternVL 3.0**: 📖 [Introduction](https://internvl.readthedocs.io/en/latest/internvl3.0/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl3.0/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html) | 📊 [Evaluation](https://internvl.readthedocs.io/en/latest/internvl3.0/evaluation.html) | 📦 [Deployment](https://internvl.readthedocs.io/en/latest/internvl3.0/deployment.html) | 🎯 [Preference Optimization](https://internvl.readthedocs.io/en/latest/internvl3.0/preference_optimization.html)
- **InternVL 2.5**: 📖 [Introduction](https://internvl.readthedocs.io/en/latest/internvl2.5/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl2.5/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html) | 📊 [Evaluation](https://internvl.readthedocs.io/en/latest/internvl2.5/evaluation.html) | 📦 [Deployment](https://internvl.readthedocs.io/en/latest/internvl2.5/deployment.html) | 🎯 [Preference Optimization](https://internvl.readthedocs.io/en/latest/internvl2.5/preference_optimization.html)
- **InternVL 2.0**: 📖 [Introduction](https://internvl.readthedocs.io/en/latest/internvl2.0/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl2.0/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html) | 📊 [Evaluation](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html) | 📦 [Deployment](https://internvl.readthedocs.io/en/latest/internvl2.0/deployment.html) | 🎯 [Preference Optimization](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html)
- **InternVL 1.5**: 📖 [Introduction](https://internvl.readthedocs.io/en/latest/internvl1.5/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl1.5/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl1.5/finetune.html) | 📊 [Evaluation](https://internvl.readthedocs.io/en/latest/internvl1.5/evaluation.html) | 📦 [Deployment](https://internvl.readthedocs.io/en/latest/internvl1.5/deployment.html)
- **InternVL 1.2**: 📖 [Introduction](https://internvl.readthedocs.io/en/latest/internvl1.2/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl1.2/quick_start.html) | ✨ [Finetune](https://internvl.readthedocs.io/en/latest/internvl1.2/finetune.html) | 📊 [Evaluation](https://internvl.readthedocs.io/en/latest/internvl1.2/evaluation.html)
- **InternVL 1.1**: 📖 [Introduction](https://internvl.readthedocs.io/en/latest/internvl1.1/introduction.html) | ⚡ [Quick Start](https://internvl.readthedocs.io/en/latest/internvl1.1/quick_start.html) | 📊 [Evaluation](https://internvl.readthedocs.io/en/latest/internvl1.1/evaluation.html)
