import torch
from internvl_hf_stage2 import \
    load_internvl_qformer as load_internvl_qformer_hf
from PIL import Image

image_size = 224
num_tokens = (image_size // 14) ** 2
model, transform, tokenizer = load_internvl_qformer_hf('xx', 'xx', 'xx', 'cuda', 'retrieval')
image = Image.open('img.png')
image = transform(image.convert('RGB')).unsqueeze(0).cuda().to(torch.bfloat16)
print(image.shape)

input_ids = tokenizer(['a photo of dog bite human', 'a photo of human bite dog']).cuda()
print('a photo of dog bite human', 'a photo of human bite dog')
print(input_ids)
model = model.to(torch.bfloat16)
model.eval()

logits_per_image, logits_per_text = model(image, input_ids)
print('logits_per_image:', logits_per_image)
print('logits_per_text:', logits_per_text)
