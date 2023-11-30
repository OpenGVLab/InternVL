import time

import torch
# if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
#     internvl_hf_stage2.modeling_qllama.LlamaAttention.forward = forward_2
#     print("replace LlamaAttention.forward with llama_flash_attn_monkey_patch.forward_2")
# else:
#     internvl_hf_stage2.modeling_qllama.LlamaAttention.forward = forward
#     print("replace LlamaAttention.forward with llama_flash_attn_monkey_patch.forward")
from internvl_hf_stage2 import \
    load_internvl_qformer as load_internvl_qformer_hf
from tqdm import tqdm

batch_size = 128
iter = 30
image_size = 448
num_tokens = (image_size // 14) ** 2
model, transform, tokenizer = load_internvl_qformer_hf('xx', 'xx', 'xx', 'cuda', 'retrieval')
image = torch.rand((batch_size, 3, image_size, image_size)).cuda().to(torch.bfloat16)
input_ids = torch.rand(batch_size, 80).long().cuda()
model = model.to(torch.bfloat16)
model.vision_model.resize_pos_embeddings(
    old_size=224,
    new_size=image_size,
    patch_size=14
)
model.eval()

print('warmup vision_model')
with torch.no_grad():
    for i in tqdm(range(iter // 10)):
        output = model.vision_model(pixel_values=image)[0]
        print(output.shape)

print('test vision_model')
torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    for i in tqdm(range(iter)):
        output = model.vision_model(pixel_values=image)[0]
torch.cuda.synchronize()
end_time = time.time()
print('time:', end_time - start_time)
print('FPS:', iter * batch_size / (end_time - start_time))

# print("warmup encode_text")
# with torch.no_grad():
#     for i in tqdm(range(iter // 10)):
#         output = model.encode_text(input_ids)
#         print(output.shape)


# print("test encode_text")
# torch.cuda.synchronize()
# start_time = time.time()
# with torch.no_grad():
#     for i in tqdm(range(iter)):
#         output = model.encode_text(input_ids)
# torch.cuda.synchronize()
# end_time = time.time()
# print("time:", end_time - start_time)
# print("FPS:", iter * batch_size / (end_time - start_time))


print('warmup encode_image')
with torch.no_grad():
    for i in tqdm(range(iter // 10)):
        output = model.encode_image(image)
        print(output.shape)

print('test encode_image')
torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    for i in tqdm(range(iter)):
        output = model.encode_image(image)
torch.cuda.synchronize()
end_time = time.time()
print('time:', end_time - start_time)
print('FPS:', iter * batch_size / (end_time - start_time))

# image = torch.rand(batch_size, num_tokens, 3200).cuda().to(torch.bfloat16)
# print("warmup encode_image_text")
# with torch.no_grad():
#     for i in tqdm(range(iter // 10)):
#         output = model.encode_image_text(image, input_ids)
#         print(output.shape)
#
# print("test encode_image_text")
# torch.cuda.synchronize()
# start_time = time.time()
# with torch.no_grad():
#     for i in tqdm(range(iter)):
#         output = model.encode_image_text(image, input_ids)
# torch.cuda.synchronize()
# end_time = time.time()
# print("time:", end_time - start_time)
# print("FPS:", iter * batch_size / (end_time - start_time))
