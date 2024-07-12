# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import base64
import json
from io import BytesIO

import requests
from PIL import Image


def get_model_list(controller_url):
    ret = requests.post(controller_url + '/refresh_all_workers')
    assert ret.status_code == 200
    ret = requests.post(controller_url + '/list_models')
    models = ret.json()['models']
    return models


def get_selected_worker_ip(controller_url, selected_model):
    ret = requests.post(controller_url + '/get_worker_address',
            json={'model': selected_model})
    worker_addr = ret.json()['address']
    return worker_addr


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


controller_url = 'http://10.140.60.209:10075'
model_list = get_model_list(controller_url)
print(f'Model list: {model_list}')

selected_model = 'InternVL2-1B'
worker_addr = get_selected_worker_ip(controller_url, selected_model)
print(f'model_name: {selected_model}, worker_addr: {worker_addr}')


# 多轮/多图对话请把数据组织成以下格式：
# send_messages = [{'role': 'system', 'content': system_message}]
# send_messages.append({'role': 'user', 'content': 'question1 to image1', 'image': [pil_image_to_base64(image)]})
# send_messages.append({'role': 'assistant', 'content': 'answer1'})
# send_messages.append({'role': 'user', 'content': 'question2 to image2', 'image': [pil_image_to_base64(image)]})
# send_messages.append({'role': 'assistant', 'content': 'answer2'})
# send_messages.append({'role': 'user', 'content': 'question3 to image1 & 2', 'image': []})

image = Image.open('image1.jpg')
print(f'Loading image, size: {image.size}')
system_message = """我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。人工智能实验室致力于原始技术创新，开源开放，共享共创，推动科技进步和产业发展。
请尽可能详细地回答用户的问题。"""
send_messages = [{'role': 'system', 'content': system_message}]
send_messages.append({'role': 'user', 'content': 'describe this image in detail', 'image': [pil_image_to_base64(image)]})

pload = {
    'model': selected_model,
    'prompt': send_messages,
    'temperature': 0.8,
    'top_p': 0.7,
    'max_new_tokens': 2048,
    'max_input_tiles': 12,
    'repetition_penalty': 1.0,
}
headers = {'User-Agent': 'InternVL-Chat Client'}
response = requests.post(worker_addr + '/worker_generate_stream',
                         headers=headers, json=pload, stream=True, timeout=10)
for chunk in response.iter_lines(decode_unicode=False, delimiter=b'\0'):
    if chunk:
        data = json.loads(chunk.decode())
        if data['error_code'] == 0:
            output = data['text'] # 这里是流式输出
        else:
            output = data['text'] + f" (error_code: {data['error_code']})"
# 完整的输出
print(output)
