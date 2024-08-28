import torch
from transformers import AutoModel, AutoTokenizer

path = 'OpenGVLab/InternVL-Chat-V1-5'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    load_in_8bit=True).eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

model.save_pretrained('release/InternVL-Chat-V1-5-Int8')
tokenizer.save_pretrained('release/InternVL-Chat-V1-5-Int8')
print('finished')
