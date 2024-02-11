import argparse
import os.path

import torch
from internvl.model.internvl_chat import InternVLChatModel

argparse = argparse.ArgumentParser()
argparse.add_argument('model_path', type=str, default='')
argparse.add_argument('output_path', type=str, default='')

args = argparse.parse_args()

model = InternVLChatModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
model = model.mlp1.to(torch.bfloat16)

ckpt = model.state_dict()
output_path = os.path.join(args.output_path, 'mlp_projector.pth')
torch.save(ckpt, output_path)
print('finished')
