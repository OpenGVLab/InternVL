# InternVL

The official implementation of

[InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](<>).

\[[Paper](<>)\]  \[[Demo](<>)\]

The exponential growth of large language models (LLMs) has opened up numerous possibilities for multi-modal AGI systems.
However, the progress in vision and vision-language foundation models, which are also critical elements of multi-modal AGI, has not kept pace with LLMs.
In this work, we design a large-scale vision-language foundation model (InternVL), which scales up the vision foundation model to 6 billion parameters and progressively aligns it with the large language model, using web-scale image-text data from various sources.
This model can be broadly applied to and achieve state-of-the-art performance on visual perception tasks such as image-level or pixel-level recognition, vision-language tasks such as zero-shot image/video classification, zero-shot image/video-text retrieval, and link with LLMs to create multi-modal dialogue systems.
We hope that our research could contribute to the development of multi-modal large models.


<img width="1204" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/bb8929c8-b653-45ef-a0fc-628a798a1a62">


## 🗓️ Schedule

- [ ] Release InternVL-Chat (our custom codebase)
- [ ] Release InternVL-Chat (LLaVA codebase)
- [ ] Release InternVL
- [ ] Release InternViT-6B

## 🏠 Overview

<img width="1204" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/d081bdcf-525a-4f84-98d8-30a0c353d6e9">

## 🎁 Performance

<img width="1204" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/c9f93b54-fdba-4a69-9341-e905376f7b9c">

## 🎫 License

This project is released under the [MIT license](LICENSE).

## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
TODO
```

## Acknowledgement

InternVL is built with [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip), [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), [EVA](https://github.com/baaivision/EVA/tree/master), [InternImage](https://github.com/OpenGVLab/InternImage), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Transformers](https://github.com/huggingface/transformers), [DINOv2](https://github.com/facebookresearch/dinov2), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm), and [LLaVA-1.5](https://github.com/haotian-liu/LLaVA). Thanks for their awesome work!
