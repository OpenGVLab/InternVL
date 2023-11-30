import json
import os.path

import torch
from internvl_hf_stage2 import build_transform
from PIL import Image
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

batch_size = 128
transform = build_transform(task='retrieval')
data_root = '/mnt/petrelfs/share_data/chenzhe1/data/coco/'
ann_path = os.path.join(data_root, 'annotations/coco_karpathy_test.json')
data = json.load(open(ann_path, 'r'))[:batch_size]
image_paths = []
captions = []
for item in tqdm(data):
    image = item['image']
    image = os.path.join(data_root, image)
    image = Image.open(image).convert('RGB')
    image = transform(image)
    image_paths.append(image)
    caption = item['caption'][0]
    captions.append(caption)

images = torch.stack(image_paths, dim=0).cuda().to(torch.bfloat16)
print('images:', images.shape)

# image_size = 224
# num_tokens = (image_size // 14) ** 2
# model, transform, tokenizer = load_internvl_qformer_hf("xx", "xx", "xx", "cuda", "retrieval")
# model = model.to(torch.bfloat16)
# model.vision_model.resize_pos_embeddings(
#     old_size=224,
#     new_size=image_size,
#     patch_size=14
# )
# model.eval()
#
# with torch.no_grad():
#     backbone_embeds, image_embeds = model.get_image_features(
#         pixel_values=images,
#         output_hidden_states=False,
#         return_dict=True,
#     )
# print("backbone_embeds:", backbone_embeds.shape)
# torch.save(backbone_embeds, "backbone_embeds.pt")
# print("image_embeds:", image_embeds.shape)
# torch.save(image_embeds, "image_embeds.pt")


backbone_embeds = torch.load('backbone_embeds.pt')[:, 1:, :].mean(dim=1)
print('backbone_embeds:', backbone_embeds.shape)
image_embeds = torch.load('image_embeds.pt').mean(dim=1)
print('image_embeds:', image_embeds.shape)

model_path = '/mnt/petrelfs/share_data/wangweiyun/share_hf/lmsys/vicuna-7b-v1.5'
vicuna = LlamaForCausalLM.from_pretrained(model_path)
vicuna.eval()

tokenizer = LlamaTokenizer.from_pretrained(model_path)

input_ids = tokenizer(captions, padding=True, return_tensors='pt').input_ids.cuda()
print('input_ids:', input_ids.shape)
input_embeddings = vicuna.get_input_embeddings().cuda()
text_embeds = input_embeddings(input_ids)
print('text_embeds:', text_embeds.shape, text_embeds.dtype)
print('before:', backbone_embeds.shape, image_embeds.shape, input_ids.shape)
for i in range(batch_size):
    backbone_embed = backbone_embeds[i]
    image_embed = image_embeds[i]
    selected = input_ids[i:i + 1]
    print(selected, selected.shape)
    # print(text_embeds[i].shape)
    # text_embed = text_embeds[i][selected]
    # print(backbone_embed.shape, image_embed.shape, text_embed.shape)
