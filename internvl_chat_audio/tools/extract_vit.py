import argparse

import torch
from internvl.model.internvl_chat import InternVLChatModel

argparse = argparse.ArgumentParser()
argparse.add_argument('model_path', type=str, default='')
argparse.add_argument('output_path', type=str, default='')

args = argparse.parse_args()

model = InternVLChatModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
model = model.vision_model.to(torch.bfloat16)

model.save_pretrained(args.output_path)
print('finished')
