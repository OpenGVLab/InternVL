import torch
from internvl_hf_stage2_v2.modeling_intern_qformer import (InternQformerConfig,
                                                           InternQformerModel)
from transformers import LlamaTokenizer

model_path = '/mnt/petrelfs/share_data/wangwenhai/internvl/internvl_huggingface/intern_clip_13b'
config = InternQformerConfig.from_pretrained(model_path)
print(config)
state_dict = {}
for i in range(6):
    temp = torch.load(f'{model_path}/pytorch_model-0000{i + 1}-of-00006.bin', 'cpu')
    state_dict.update(temp)

new_state_dict = {}
for k, v in state_dict.items():
    if 'logit_scale' in k:
        continue
    if 'language_model.' in k:
        new_state_dict[k.replace('language_model.', 'qformer.')] = v
    else:
        new_state_dict[k] = v

model = InternQformerModel(config)
message = model.load_state_dict(new_state_dict, strict=False)
print(message)

model.save_pretrained('./intern_qformer_13b_v2')
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained('./intern_qformer_13b_v2')
