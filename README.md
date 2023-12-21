# InternVL

The official implementation of

[InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](<>).

\[[Paper](<>)\]  \[[Demo](<>)\]

The exponential growth of large language models (LLMs) has opened up numerous possibilities for multi-modal AGI systems.
However, the progress in vision and vision-language foundation models, which are also critical elements of multi-modal AGI, has not kept pace with LLMs.
In this work, we design a large-scale vision-language foundation model (InternVL), which scales up the vision foundation model to 6 billion parameters and progressively aligns it with the large language model, using web-scale image-text data from various sources.
This model can be broadly applied to and achieve state-of-the-art performance on visual perception tasks such as image-level or pixel-level recognition, vision-language tasks such as zero-shot image/video classification, zero-shot image/video-text retrieval, and link with LLMs to create multi-modal dialogue systems.
We hope that our research could contribute to the development of multi-modal large models.

## 🗓️ Schedule

- [ ] Release InternVL-Chat
- [ ] Release InternVL-14B
- [ ] Release InternViT-6B

## 🏠 Overview

<img width="971" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/7922e66d-f969-4329-9f42-4dc9cb4ef46a">

## 🎫 License

This project is released under the [MIT license](LICENSE).

## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
TODO
```

## Acknowledgement

InternVL is built with [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip), [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), [EVA](https://github.com/baaivision/EVA/tree/master), [InternImage](https://github.com/OpenGVLab/InternImage), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Transformers](https://github.com/huggingface/transformers), [Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm), and [LLaVA-1.5](https://github.com/haotian-liu/LLaVA). Thanks for their awesome work!
