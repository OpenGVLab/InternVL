# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import json
import math
import threading
import time
import uuid
from functools import partial
from io import BytesIO
from threading import Thread

import requests
import torch
import torchvision.transforms as T
import uvicorn
from constants import IMAGENET_MEAN, IMAGENET_STD, WORKER_HEART_BEAT_INTERVAL
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer
from utils import build_logger, pretty_print_semaphore, server_error_msg

worker_id = str(uuid.uuid4())[:6]
logger = build_logger('model_worker', f'model_worker_{worker_id}.log')
global_counter = 0
model_semaphore = None


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def split_model(model_name, vit_alpha=0.5):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL-Chat-V1-1': 40, 'InternVL-Chat-V1-2': 60, 'InternVL-Chat-V1-2-Plus': 60,
        'Mini-InternVL-2B-V1-5': 24, 'Mini-InternVL-4B-V1-5': 32, 'InternVL-Chat-V1-5': 48,
        'InternVL2-8B': 32, 'InternVL2-26B': 48,  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80,
        'InternVL2-78B': 80, 'InternVL2-Pro': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - vit_alpha))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vit_alpha))
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


class ModelWorker:
    def __init__(self, controller_addr, worker_addr, worker_id, model_path, model_name,
                 load_8bit, device, context_len=8192):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith('/'):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split('/')
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + '_' + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        logger.info(f'Loading the model {self.model_name} on worker {worker_id} ...')

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        tokens_to_keep = ['<box>', '</box>', '<ref>', '</ref>']
        tokenizer.additional_special_tokens = [item for item in tokenizer.additional_special_tokens if item not in tokens_to_keep]
        self.tokenizer = tokenizer

        if device == 'auto':
            device_map = split_model(self.model_name)
            self.model = AutoModel.from_pretrained(
                model_path,
                load_in_8bit=load_8bit,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True).eval()
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                load_in_8bit=load_8bit,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).eval()
        if not load_8bit and not device == 'auto':
            self.model = self.model.cuda()
        self.load_8bit = load_8bit
        self.device = device
        self.model_path = model_path
        self.image_size = self.model.config.force_image_size
        self.context_len = context_len
        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker, args=(self,))
        self.heart_beat_thread.start()

    def reload_model(self):
        del self.model
        torch.cuda.empty_cache()
        if self.device == 'auto':
            device_map = split_model(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_path,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True).eval()
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).eval()
        if not self.load_8bit and not self.device == 'auto':
            self.model = self.model.cuda()

    def register_to_controller(self):
        logger.info('Register to controller')

        url = self.controller_addr + '/register_worker'
        data = {
            'worker_name': self.worker_addr,
            'check_heart_beat': True,
            'worker_status': self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f'Send heart beat. Models: {[self.model_name]}. '
                    f'Semaphore: {pretty_print_semaphore(model_semaphore)}. '
                    f'global_counter: {global_counter}')

        url = self.controller_addr + '/receive_heart_beat'

        while True:
            try:
                ret = requests.post(url, json={
                    'worker_name': self.worker_addr,
                    'queue_length': self.get_queue_length()}, timeout=5)
                exist = ret.json()['exist']
                break
            except requests.exceptions.RequestException as e:
                logger.error(f'heart beat error: {e}')
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            'model_names': [self.model_name],
            'speed': 1,
            'queue_length': self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        system_message = params['prompt'][0]['content']
        send_messages = params['prompt'][1:]
        max_input_tiles = params['max_input_tiles']
        temperature = params['temperature']
        top_p = params['top_p']
        max_new_tokens = params['max_new_tokens']
        repetition_penalty = params['repetition_penalty']
        do_sample = True if temperature > 0.0 else False

        global_image_cnt = 0
        history, pil_images, max_input_tile_list = [], [], []
        for message in send_messages:
            if message['role'] == 'user':
                prefix = ''
                if 'image' in message:
                    max_input_tile_temp = []
                    for image_str in message['image']:
                        pil_images.append(load_image_from_base64(image_str))
                        prefix += f'Image-{global_image_cnt + 1}: <image>\n'
                        global_image_cnt += 1
                        max_input_tile_temp.append(max(1, max_input_tiles // len(message['image'])))
                    if len(max_input_tile_temp) > 0:
                        max_input_tile_list.append(max_input_tile_temp)
                content = prefix + message['content']
                history.append([content, ])
            else:
                history[-1].append(message['content'])
        question, history = history[-1][0], history[:-1]

        if global_image_cnt == 1:
            question = question.replace('Image-1: <image>\n', '<image>\n')
            history = [[item[0].replace('Image-1: <image>\n', '<image>\n'), item[1]] for item in history]

        # Create a new list to store processed sublists
        flattened_list = []
        # Iterate through all but the last sublist in max_input_tile_list and process them
        for sublist in max_input_tile_list[:-1]:
            processed_sublist = [1] * len(sublist)  # Change each element in the sublist to 1
            flattened_list.extend(processed_sublist)  # Flatten the processed sublist and add to the new list
        # If max_input_tile_list is not empty, add the last sublist to the new list
        if max_input_tile_list:
            flattened_list.extend(max_input_tile_list[-1])
        max_input_tile_list = flattened_list
        assert len(max_input_tile_list) == len(pil_images), 'The number of max_input_tile_list and pil_images should be the same.'

        old_system_message = self.model.system_message
        self.model.system_message = system_message
        image_tiles, num_patches_list = [], []
        transform = build_transform(input_size=self.image_size)
        if len(pil_images) > 0:
            for current_max_input_tiles, pil_image in zip(max_input_tile_list, pil_images):
                if self.model.config.dynamic_image_size:
                    tiles = dynamic_preprocess(
                        pil_image, image_size=self.image_size, max_num=current_max_input_tiles,
                        use_thumbnail=self.model.config.use_thumbnail)
                else:
                    tiles = [pil_image]
                num_patches_list.append(len(tiles))
                image_tiles += tiles
            pixel_values = [transform(item) for item in image_tiles]
            pixel_values = torch.stack(pixel_values).to(self.model.device, dtype=torch.bfloat16)
            logger.info(f'Split images to {pixel_values.shape}')
        else:
            pixel_values = None

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
        generation_config = dict(
            num_beams=1,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_length=self.context_len,
            top_p=top_p,
            streamer=streamer,
        )
        logger.info(f'Generation config: {generation_config}')

        thread = Thread(target=self.model.chat, kwargs=dict(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=question,
            history=history,
            return_history=False,
            generation_config=generation_config,
        ))
        thread.start()

        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(self.model.conv_template.sep):
                generated_text = generated_text[:-len(self.model.conv_template.sep)]
            yield json.dumps({'text': generated_text, 'error_code': 0}).encode() + b'\0'
        logger.info(f'max_input_tile_list: {max_input_tile_list}, history: {history}, '
                    f'question: {question}, answer: {generated_text}')
        self.model.system_message = old_system_message

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print('Caught ValueError:', e)
            ret = {
                'text': server_error_msg,
                'error_code': 1,
            }
            yield json.dumps(ret).encode() + b'\0'
        except torch.cuda.CudaError as e:
            print('Caught torch.cuda.CudaError:', e)
            ret = {
                'text': server_error_msg,
                'error_code': 1,
            }
            yield json.dumps(ret).encode() + b'\0'
        except Exception as e:
            print('Caught Unknown Error', e)
            ret = {
                'text': server_error_msg,
                'error_code': 1,
            }
            yield json.dumps(ret).encode() + b'\0'


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post('/worker_generate_stream')
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post('/worker_get_status')
async def get_status(request: Request):
    return worker.get_status()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=21002)
    parser.add_argument('--worker-address', type=str, default='http://localhost:21002')
    parser.add_argument('--controller-address', type=str, default='http://localhost:21001')
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--limit-model-concurrency', type=int, default=5)
    parser.add_argument('--stream-interval', type=int, default=1)
    parser.add_argument('--load-8bit', action='store_true')
    args = parser.parse_args()
    logger.info(f'args: {args}')

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.model_path,
                         args.model_name,
                         args.load_8bit,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
