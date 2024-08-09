import argparse

import torch
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer

argparse = argparse.ArgumentParser()
argparse.add_argument('input_path', type=str, help='Path to the input model')
argparse.add_argument('output_path', type=str, help='Path to the output model')
args = argparse.parse_args()

print('Loading model...')
model = InternVLChatModel.from_pretrained(
    args.input_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval()
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)

if model.config.use_backbone_lora:
    model.vision_model.merge_and_unload()
    model.vision_model = model.vision_model.model
    model.config.use_backbone_lora = 0
if model.config.use_llm_lora:
    model.language_model.merge_and_unload()
    model.language_model = model.language_model.model
    model.config.use_llm_lora = 0

print('Saving model...')
model.save_pretrained(args.output_path)
print('Saving tokenizer...')
tokenizer.save_pretrained(args.output_path)
print('Done!')
