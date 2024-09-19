import argparse

import torch
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer

argparse = argparse.ArgumentParser()
argparse.add_argument('model_path', type=str, default='')
argparse.add_argument('output_path', type=str, default='')
argparse.add_argument('force_image_size', type=int, default=448)

args = argparse.parse_args()

model = InternVLChatModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                         new_size=args.force_image_size,
                                         patch_size=14)
model.config.vision_config.image_size = args.force_image_size
model.config.force_image_size = args.force_image_size

model.save_pretrained(args.output_path)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.save_pretrained(args.output_path)
print('finished')
