# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import argparse
import base64
import datetime
import hashlib
import json
import os
import random
import re
import sys
# from streamlit_js_eval import streamlit_js_eval
from functools import partial
from io import BytesIO

import cv2
import numpy as np
import requests
import streamlit as st
from constants import LOGDIR, server_error_msg
from library import Library
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_select import image_select

custom_args = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--controller_url', type=str, default='http://10.140.60.209:10075', help='url of the controller')
parser.add_argument('--sd_worker_url', type=str, default='http://0.0.0.0:40006', help='url of the stable diffusion worker')
parser.add_argument('--max_image_limit', type=int, default=4, help='maximum number of images')
args = parser.parse_args(custom_args)
controller_url = args.controller_url
sd_worker_url = args.sd_worker_url
max_image_limit = args.max_image_limit
print('args:', args)


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f'{t.year}-{t.month:02d}-{t.day:02d}-conv.json')
    return name


def get_model_list():
    ret = requests.post(controller_url + '/refresh_all_workers')
    assert ret.status_code == 200
    ret = requests.post(controller_url + '/list_models')
    models = ret.json()['models']
    models = [item for item in models if 'InternVL2-Det' not in item and 'InternVL2-Gen' not in item]
    return models


def load_upload_file_and_show():
    if uploaded_files is not None:
        images, filenames = [], []
        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            images.append(img)
        with upload_image_preview.container():
            Library(images)

        image_hashes = [hashlib.md5(image.tobytes()).hexdigest() for image in images]
        for image, hash in zip(images, image_hashes):
            t = datetime.datetime.now()
            filename = os.path.join(LOGDIR, 'serve_images', f'{t.year}-{t.month:02d}-{t.day:02d}', f'{hash}.jpg')
            filenames.append(filename)
            if not os.path.isfile(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                image.save(filename)
    return images, filenames


def get_selected_worker_ip():
    ret = requests.post(controller_url + '/get_worker_address',
            json={'model': selected_model})
    worker_addr = ret.json()['address']
    return worker_addr


def save_chat_history():
    messages = st.session_state.messages
    new_messages = []
    for message in messages:
        new_message = {'role': message['role'], 'content': message['content']}
        if 'filenames' in message:
            new_message['filenames'] = message['filenames']
        new_messages.append(new_message)
    if len(new_messages) > 0:
        fout = open(get_conv_log_filename(), 'a')
        data = {
            'type': 'chat',
            'model': selected_model,
            'messages': new_messages,
        }
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
        fout.close()


def generate_response(messages):
    send_messages = [{'role': 'system', 'content': system_message_default + '\n\n' + persona_rec}]
    for message in messages:
        if message['role'] == 'user':
            user_message = {'role': 'user', 'content': message['content']}
            if 'image' in message and len('image') > 0:
                user_message['image'] = []
                for image in message['image']:
                    user_message['image'].append(pil_image_to_base64(image))
            send_messages.append(user_message)
        else:
            send_messages.append({'role': 'assistant', 'content': message['content']})
    pload = {
        'model': selected_model,
        'prompt': send_messages,
        'temperature': float(temperature),
        'top_p': float(top_p),
        'max_new_tokens': max_length,
        'max_input_tiles': max_input_tiles,
        'repetition_penalty': float(repetition_penalty),
    }
    worker_addr = get_selected_worker_ip()
    headers = {'User-Agent': 'InternVL-Chat Client'}
    placeholder, output = st.empty(), ''
    try:
        response = requests.post(worker_addr + '/worker_generate_stream',
                                 headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=True, delimiter=b'\0'):
            if chunk:
                data = json.loads(chunk.decode())
                if data['error_code'] == 0:
                    output = data['text']
                    # Phi3-3.8B will produce abnormal `�` output
                    if '4B' in selected_model and '�' in output[-2:]:
                        output = output.replace('�', '')
                        break
                    placeholder.markdown(output + '▌')
                else:
                    output = data['text'] + f" (error_code: {data['error_code']})"
                    placeholder.markdown(output)
        if ('\[' in output and '\]' in output) or ('\(' in output and '\)' in output):
            output = output.replace('\[', '$').replace('\]', '$').replace('\(', '$').replace('\)', '$')
        placeholder.markdown(output)
    except requests.exceptions.RequestException as e:
        placeholder.markdown(server_error_msg)
    return output


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def clear_chat_history():
    st.session_state.messages = []
    st.session_state['image_select'] = -1


def clear_file_uploader():
    st.session_state.uploader_key += 1
    st.rerun()


def combined_func(func_list):
    for func in func_list:
        func()


def show_one_or_multiple_images(message, total_image_num, is_input=True):
    if 'image' in message:
        if is_input:
            total_image_num = total_image_num + len(message['image'])
            if lan == 'English':
                if len(message['image']) == 1 and total_image_num == 1:
                    label = f"(In this conversation, {len(message['image'])} image was uploaded, {total_image_num} image in total)"
                elif len(message['image']) == 1 and total_image_num > 1:
                    label = f"(In this conversation, {len(message['image'])} image was uploaded, {total_image_num} images in total)"
                else:
                    label = f"(In this conversation, {len(message['image'])} images were uploaded, {total_image_num} images in total)"
            else:
                label = f"(在本次对话中，上传了{len(message['image'])}张图片，总共上传了{total_image_num}张图片)"
        upload_image_preview = st.empty()
        with upload_image_preview.container():
            Library(message['image'])
        if is_input and len(message['image']) > 0:
            st.markdown(label)


def find_bounding_boxes(response):
    pattern = re.compile(r'<ref>\s*(.*?)\s*</ref>\s*<box>\s*(\[\[.*?\]\])\s*</box>')
    matches = pattern.findall(response)
    results = []
    for match in matches:
        results.append((match[0], eval(match[1])))
    returned_image = None
    for message in st.session_state.messages:
        if message['role'] == 'user' and 'image' in message and len(message['image']) > 0:
            last_image = message['image'][-1]
            width, height = last_image.size
            returned_image = last_image.copy()
            draw = ImageDraw.Draw(returned_image)
    for result in results:
        line_width = max(1, int(min(width, height) / 200))
        random_color = (random.randint(0, 128), random.randint(0, 128), random.randint(0, 128))
        category_name, coordinates = result
        coordinates = [(float(x[0]) / 1000, float(x[1]) / 1000, float(x[2]) / 1000, float(x[3]) / 1000) for x in coordinates]
        coordinates = [(int(x[0] * width), int(x[1] * height), int(x[2] * width), int(x[3] * height)) for x in coordinates]
        for box in coordinates:
            draw.rectangle(box, outline=random_color, width=line_width)
            font = ImageFont.truetype('static/SimHei.ttf', int(20 * line_width / 2))
            text_size = font.getbbox(category_name)
            text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]
            text_position = (box[0], max(0, box[1] - text_height))
            draw.rectangle(
                [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                fill=random_color
            )
            draw.text(text_position, category_name, fill='white', font=font)
    return returned_image if len(matches) > 0 else None


def query_image_generation(response, sd_worker_url, timeout=15):
    sd_worker_url = f'{sd_worker_url}/generate_image/'
    pattern = r'```drawing-instruction\n(.*?)\n```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        payload = {'caption': match.group(1)}
        print('drawing-instruction:', payload)
        response = requests.post(sd_worker_url, json=payload, timeout=timeout)
        response.raise_for_status()  # 检查HTTP请求是否成功
        image = Image.open(BytesIO(response.content))
        return image
    else:
        return None


def regenerate():
    st.session_state.messages = st.session_state.messages[:-1]
    st.rerun()


logo_code = """
<svg width="1700" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color: red; stop-opacity: 1" />
      <stop offset="100%" style="stop-color: orange; stop-opacity: 1" />
    </linearGradient>
  </defs>
  <text x="000" y="160" font-size="180" font-weight="bold" fill="url(#gradient1)" style="font-family: Arial, sans-serif;">
    InternVL2 Demo
  </text>
</svg>
"""

# App title
st.set_page_config(page_title='InternVL2')

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

system_message_default = '我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'

system_message_editable = '请尽可能详细地回答用户的问题。'

# Replicate Credentials
with st.sidebar:
    model_list = get_model_list()
    # "[![Open in GitHub](https://github.com/codespaces/badge.svg)](https://github.com/OpenGVLab/InternVL)"
    lan = st.selectbox('#### Language / 语言', ['English', '中文'], on_change=st.rerun,
                       help='This is only for switching the UI language. 这仅用于切换UI界面的语言。')
    if lan == 'English':
        st.logo(logo_code, link='https://github.com/OpenGVLab/InternVL', icon_image=logo_code)
        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a InternVL2 chat model', model_list, key='selected_model',
                                              on_change=clear_chat_history,
                                              help='Due to the limited GPU resources with public IP addresses, we can currently only deploy models up to a maximum of 26B.')
        with st.expander('🤖 System Prompt'):
            persona_rec = st.text_area('System Prompt', value=system_message_editable,
                                       help='System prompt is a pre-defined message used to instruct the assistant at the beginning of a conversation.',
                                       height=200)
        with st.expander('🔥 Advanced Options'):
            temperature = st.slider('temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.95, step=0.05)
            repetition_penalty = st.slider('repetition_penalty', min_value=1.0, max_value=1.5, value=1.1, step=0.02)
            max_length = st.slider('max_new_token', min_value=0, max_value=4096, value=1024, step=128)
            max_input_tiles = st.slider('max_input_tiles (control image resolution)', min_value=1, max_value=24,
                                        value=12, step=1)
        upload_image_preview = st.empty()
        uploaded_files = st.file_uploader('Upload files', accept_multiple_files=True,
                                          type=['png', 'jpg', 'jpeg', 'webp'],
                                          help='You can upload multiple images (max to 4) or a single video.',
                                          key=f'uploader_{st.session_state.uploader_key}',
                                          on_change=st.rerun)
        uploaded_pil_images, save_filenames = load_upload_file_and_show()
        todo_list = st.sidebar.selectbox('Our to-do list', ['👏This is our to-do list',
                                                            '1. Support for video uploads',
                                                            '2. Support for PDF uploads',
                                                            '3. Write a usage document'], key='todo_list',
                                         help='Here are some features we plan to support in the future.')
    else:
        st.subheader('模型和参数')
        selected_model = st.sidebar.selectbox('选择一个 InternVL2 对话模型', model_list, key='selected_model',
                                              on_change=clear_chat_history,
                                              help='由于有限的公网GPU资源，我们暂时只能部署到最大参数26B的模型。')
        with st.expander('🤖 系统提示'):
            persona_rec = st.text_area('系统提示', value=system_message_editable,
                                       help='系统提示是在对话开始时用于指示助手的预定义消息。',
                                       height=200)
        with st.expander('🔥 高级选项'):
            temperature = st.slider('temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            top_p = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.95, step=0.05)
            repetition_penalty = st.slider('重复惩罚', min_value=1.0, max_value=1.5, value=1.1, step=0.02)
            max_length = st.slider('最大输出长度', min_value=0, max_value=4096, value=1024, step=128)
            max_input_tiles = st.slider('最大图像块数 (控制图像分辨率)', min_value=1, max_value=24, value=12, step=1)
        upload_image_preview = st.empty()
        uploaded_files = st.file_uploader('上传文件', accept_multiple_files=True,
                                          type=['png', 'jpg', 'jpeg', 'webp'],
                                          help='你可以上传多张图像（最多4张）或者一个视频。',
                                          key=f'uploader_{st.session_state.uploader_key}',
                                          on_change=st.rerun)
        uploaded_pil_images, save_filenames = load_upload_file_and_show()
        todo_list = st.sidebar.selectbox('我们的待办事项', ['👏这里是我们的待办事项', '1. 支持上传视频',
                                                     '2. 支持上传 PDF 文档', '3. 写一个使用文档'], key='todo_list',
                                         help='这是我们计划要支持的一些功能。')

gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
</style>
<div class="gradient-text">InternVL2</div>
"""
if lan == 'English':
    st.markdown(gradient_text_html, unsafe_allow_html=True)
    st.caption('Expanding Performance Boundaries of Open-Source Multimodal Large Language Models')
else:
    st.markdown(gradient_text_html.replace('InternVL2', '书生·万象'), unsafe_allow_html=True)
    st.caption('扩展开源多模态大语言模型的性能边界')

# Store LLM generated responses
if 'messages' not in st.session_state.keys():
    clear_chat_history()

gallery_placeholder = st.empty()
with gallery_placeholder.container():
    examples = ['gallery/prod_9.jpg', 'gallery/astro_on_unicorn.png',
                'gallery/prod_12.png', 'gallery/prod_en_17.png',
                'gallery/prod_4.png', 'gallery/cheetah.png', 'gallery/prod_1.jpeg']
    images = [Image.open(image) for image in examples]
    if lan == 'English':
        captions = ["What's at the far end of the image?",
                    'Could you help me draw a picture like this one?',
                    'What are the consequences of the easy decisions shown in this image?',
                    "I'm on a diet, but I really want to eat them.",
                    'Is this a real plant? Analyze the reasons.',
                    'Detect the <ref>the middle leopard</ref> in the image with its bounding box.',
                    'Please identify and label all objects in the following image.']
    else:
        captions = ['画面最远处是什么?',
                    '请画一张类似这样的画',
                    '这张图上 easy decisions 导致了什么后果?',
                    '我在减肥，但我真的很想吃这个。',
                    '这是真的植物吗？分析原因',
                    '在以下图像中进行目标检测，并标出所有物体。',
                    '这幅图的氛围如何？']
    img_idx = image_select(
        label='',
        images=images,
        captions=captions,
        use_container_width=True,
        index=-1,
        return_value='index',
        key='image_select'
    )
    if lan == 'English':
        st.caption(
            'Note: For non-commercial research use only. AI responses may contain errors. Users should not spread or allow others to spread hate speech, violence, pornography, or fraud-related harmful information.')
    else:
        st.caption('注意：仅限非商业研究使用。用户应不传播或允许他人传播仇恨言论、暴力、色情内容或与欺诈相关的有害信息。')
    if img_idx != -1 and len(st.session_state.messages) == 0 and selected_model is not None:
        gallery_placeholder.empty()
        st.session_state.messages.append({'role': 'user', 'content': captions[img_idx], 'image': [images[img_idx]],
                                          'filenames': [examples[img_idx]]})
        st.rerun()  # Fixed an issue where examples were not emptied

if len(st.session_state.messages) > 0:
    gallery_placeholder.empty()

# Display or clear chat messages
total_image_num = 0
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        show_one_or_multiple_images(message, total_image_num, is_input=message['role'] == 'user')
        if 'image' in message and message['role'] == 'user':
            total_image_num += len(message['image'])

input_disable_flag = (len(model_list) == 0) or total_image_num + len(uploaded_files) > max_image_limit
if lan == 'English':
    st.sidebar.button('Clear Chat History',
                      on_click=partial(combined_func, func_list=[clear_chat_history, clear_file_uploader]))
    if input_disable_flag:
        prompt = st.chat_input('Too many images have been uploaded. Please clear the history.',
                               disabled=input_disable_flag)
    else:
        prompt = st.chat_input('Send messages to InternVL', disabled=input_disable_flag)
else:
    st.sidebar.button('清空聊天记录', on_click=partial(combined_func, func_list=[clear_chat_history, clear_file_uploader]))
    if input_disable_flag:
        prompt = st.chat_input('输入的图片太多了，请清空历史记录。', disabled=input_disable_flag)
    else:
        prompt = st.chat_input('给 “InternVL” 发送消息', disabled=input_disable_flag)

alias_instructions = {
    '目标检测': '在以下图像中进行目标检测，并标出所有物体。',
    '检测': '在以下图像中进行目标检测，并标出所有物体。',
    'object detection': 'Please identify and label all objects in the following image.',
    'detection': 'Please identify and label all objects in the following image.'
}

if prompt:
    prompt = alias_instructions[prompt] if prompt in alias_instructions else prompt
    gallery_placeholder.empty()
    image_list = uploaded_pil_images
    st.session_state.messages.append(
        {'role': 'user', 'content': prompt, 'image': image_list, 'filenames': save_filenames})
    with st.chat_message('user'):
        st.write(prompt)
        show_one_or_multiple_images(st.session_state.messages[-1], total_image_num, is_input=True)
    if image_list:
        clear_file_uploader()

# Generate a new response if last message is not from assistant
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            if not prompt:
                prompt = st.session_state.messages[-1]['content']
            response = generate_response(st.session_state.messages)
            message = {'role': 'assistant', 'content': response}
        with st.spinner('Drawing...'):
            if '<ref>' in response:
                has_returned_image = find_bounding_boxes(response)
                message['image'] = [has_returned_image] if has_returned_image else []
            if '```drawing-instruction' in response:
                has_returned_image = query_image_generation(response, sd_worker_url=sd_worker_url)
                message['image'] = [has_returned_image] if has_returned_image else []
            st.session_state.messages.append(message)
            show_one_or_multiple_images(message, total_image_num, is_input=False)

if len(st.session_state.messages) > 0:
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1.3])
    text1 = 'Clear Chat History' if lan == 'English' else '清空聊天记录'
    text2 = 'Regenerate' if lan == 'English' else '重新生成'
    text3 = 'Copy' if lan == 'English' else '复制回答'
    with col1:
        st.button(text1, on_click=partial(combined_func, func_list=[clear_chat_history, clear_file_uploader]),
                  key='clear_chat_history_button')
    with col2:
        st.button(text2, on_click=regenerate, key='regenerate_button')

print(st.session_state.messages)
save_chat_history()
