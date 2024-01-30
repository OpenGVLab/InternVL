import argparse

import torch
from internvl.model.internvl_chat import InternVLChatModel
from transformers import LlamaForCausalLM, LlamaTokenizer

argparse = argparse.ArgumentParser()
argparse.add_argument('model_path', type=str, default='')
argparse.add_argument('llm_path', type=str, default='')

args = argparse.parse_args()

if args.model_path[-1] == '/':
    args.model_path = args.model_path[:-1]

model = InternVLChatModel.from_pretrained(args.model_path)

llm = LlamaForCausalLM.from_pretrained(args.llm_path)
tokenizer = LlamaTokenizer.from_pretrained(args.llm_path)
model.language_model = llm
model.config.llm_config = llm.config
model.to(torch.bfloat16)

output_path = args.model_path + '_replace_llm'
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print('finished')
