"""
仅支持单轮speech 对话的测试
"""
import torch
from transformers import AutoTokenizer
from internvl.model.audio.processing_whisper import WhisperProcessor
from internvl.model.internvl_chat import InternVLChatAudioConfig,InternVLChatAudioModel

from internvl.train.dataset_audio import *
from internvl.model.internvl_chat.modeling_internvl_audio import Chat, single_qa


def load_model(model_path, ps_version='v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, add_eos_token=False, trust_remote_code=True)
    config = InternVLChatAudioConfig.from_pretrained(model_path)
    model = InternVLChatAudioModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda', config=config)
    model = model.eval()
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    audio_context_token_id = tokenizer.convert_tokens_to_ids(AUDIO_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    model.audio_context_token_id = audio_context_token_id
    model.ps_version = ps_version

    return model, tokenizer

model_path = 'OpenGVLab/InternOmni'
dynamic_image_size = True
max_dynamic_patch = 12
audio_processor = WhisperProcessor.from_pretrained(model_path)
model, tokenizer = load_model(model_path)
image_size = model.config.force_image_size or model.config.vision_config.image_size
use_thumbnail = model.config.use_thumbnail
device = "cuda" if torch.cuda.is_available() else "cpu"
chat = Chat(model, tokenizer, audio_processor, device,
        dynamic_image_size=dynamic_image_size,
        max_dynamic_patch=max_dynamic_patch)

# question = '这段话的文本是什么？'
image_path = './example.jpg'
audio_path = './example.wav'
answer = single_qa(chat=chat, question=None, img_path=image_path, audio_path=audio_path)
print(answer)