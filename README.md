# InternVL

The official implementation of

[InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](./paper.pdf).

\[[Paper](./paper.pdf)\]  \[[Demo](<>)\]

## Hightlights

- InternViT-6B is scaled up to 6B parameters, which can be **a good alternative to the ViT-22B**.
- QLLaMA acts as a strong text encoder, or language-guided image encoder, **supporting multiple languages**.
- State-of-the-art performance on 32 generic visual-linguistic benchmarks, including:
  - Zero-Shot Capability: Image/Video Classification, Image/Video-Text Retrieval, Image Captioning.
  - It can act as a strong visual feature extractor to create multi-modal dialogue systems.

<img width="1204" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/bb8929c8-b653-45ef-a0fc-628a798a1a62">

## Schedule

- [ ] Release InternVL-Chat (our codebase)
- [ ] Release InternVL-Chat (LLaVA codebase)
- [x] Release InternVL
- [x] Release InternViT-6B

## Overview

<img width="1204" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/d081bdcf-525a-4f84-98d8-30a0c353d6e9">

## Performance

<img width="1204" alt="image" src="https://github.com/OpenGVLab/InternVL/assets/23737120/c9f93b54-fdba-4a69-9341-e905376f7b9c">

## Model Weights

| model name   | type        | download                                                                                   |
| ------------ | ----------- | ------------------------------------------------------------------------------------------ |
| InternViT-6B | pytorch     | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/intern_vit_6b_224px.pth)  |
| InternViT-6B | huggingface | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternViT-6B-224px)                          |
| InternVL-C   | pytorch     | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL/blob/main/internvl_c_13b_224px.pth) |
| InternVL-C/G | huggingface | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVL-14B-224px)                          |

## Inference (Huggingface Version)

InternViT-6B

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

InternVL-C & InternVL-G

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

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:TODO},
  year={2023}
}
```

## Acknowledgement

InternVL is built with reference to the code of the following projects: [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip), [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark), [EVA](https://github.com/baaivision/EVA/tree/master), [InternImage](https://github.com/OpenGVLab/InternImage), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Transformers](https://github.com/huggingface/transformers), [DINOv2](https://github.com/facebookresearch/dinov2), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [Qwen-VL](https://github.com/QwenLM/Qwen-VL/tree/master/eval_mm), and [LLaVA-1.5](https://github.com/haotian-liu/LLaVA). Thanks for their awesome work!
