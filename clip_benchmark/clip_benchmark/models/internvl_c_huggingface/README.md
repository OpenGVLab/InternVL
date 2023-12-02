# InternVisionModel

```python
from internvl_huggingface import InternVisionModel

model_path = "/mnt/petrelfs/share_data/wangwenhai/internvl/internvl_huggingface/intern_clip_vit_6b"
model = InternVisionModel.from_pretrained(model_path).half().cuda()
pixel_values = torch.rand(1, 3, 224, 224).half().cuda()
vision_outputs = model(pixel_values=pixel_values)
last_hidden_state = vision_outputs.last_hidden_state
```

# InternCLIPModel

```python
from internvl_huggingface import InternCLIPModel
from transformers import LlamaTokenizer

model_path = "/mnt/petrelfs/share_data/wangwenhai/internvl/internvl_huggingface/intern_clip_13b"
model = InternCLIPModel.from_pretrained(model_path).half().cuda()
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_eos_token = True
texts = ["apple", "cat", "dog"]
tokenized = tokenizer(["summarize:"+item for item in texts], return_tensors="pt")
input_ids = tokenized.input_ids.cuda()
attention_mask = tokenized.attention_mask.cuda()
pixel_values = torch.rand(1, 3, 224, 224).half().cuda()
outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
logits_per_image = outputs.logits_per_image
logits_per_text = outputs.logits_per_text
```
