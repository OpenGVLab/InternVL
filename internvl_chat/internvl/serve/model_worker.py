"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import threading
import time
import uuid
from functools import partial
from threading import Thread

import requests
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse
from internvl.train.dataset import dynamic_preprocess
from transformers import (AutoTokenizer, CLIPImageProcessor,
                          TextIteratorStreamer)

from ..model.internvl_chat import InternVLChatModel
from .constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                        DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_TOKEN,
                        IMAGE_TOKEN_INDEX, WORKER_HEART_BEAT_INTERVAL)
from .mm_utils import (KeywordsStoppingCriteria, load_image_from_base64,
                       process_images, tokenizer_image_token)
from .utils import build_logger, pretty_print_semaphore, server_error_msg

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger('model_worker', f'model_worker_{worker_id}.log')
global_counter = 0

model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device):
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if device == 'auto':
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            # This can make distributed deployment work properly, wonder why
            self.model = InternVLChatModel.from_pretrained(
                model_path, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map='auto').eval()
        else:
            self.model = InternVLChatModel.from_pretrained(
                model_path, load_in_8bit=load_8bit, torch_dtype=torch.float16).eval()
        if not load_8bit and not device == 'auto':
            self.model = self.model.cuda()
        self.image_size = self.model.config.force_image_size
        self.image_processor = CLIPImageProcessor(
            crop_size=self.image_size, do_center_crop=True, do_normalize=True, do_resize=True,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size=self.image_size
        )
        self.context_len = 12800
        self.is_multimodal = True

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

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
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params['prompt']
        max_input_tiles = params['max_input_tiles']
        logger.info(f'max_input_tiles: {max_input_tiles}')
        ori_prompt = prompt
        images = params.get('images', None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError('Number of images does not match number of <image> tokens in prompt')
                logger.info(f'dynamic_image_size: {model.config.dynamic_image_size}')
                logger.info(f'use_thumbnail: {model.config.use_thumbnail}')
                images = [load_image_from_base64(image) for image in images]
                if model.config.dynamic_image_size:
                    images = dynamic_preprocess(
                        images[0], image_size=self.image_size, max_num=max_input_tiles,
                        use_thumbnail=model.config.use_thumbnail)
                images = [item.resize((self.image_size, self.image_size)) for item in images]
                logger.info(f'Resize images to {self.image_size}x{self.image_size}')
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)
                # images = torch.concat(images)
                logger.info(f'Split images to {images.shape}')

                replace_token = DEFAULT_IMAGE_TOKEN
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                logger.info(prompt)
                num_image_tokens = model.num_image_token * images.size(0)
                model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)
            else:
                images = None
            image_args = {'pixel_values': images}
        else:
            images = None
            image_args = {}

        temperature = float(params.get('temperature', 1.0))
        top_p = float(params.get('top_p', 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 16384)
        max_new_tokens = int(params.get('max_new_tokens', 1024))
        stop_str = params.get('stop', None)
        do_sample = True if temperature > 0.001 else False
        logger.info(f'num_image_tokens: {num_image_tokens}')
        logger.info(f'stop_str: {stop_str}')
        eos_token_id = tokenizer.convert_tokens_to_ids(stop_str)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, num_image_tokens, return_tensors='pt').unsqueeze(0).cuda()
        input_ids[input_ids==IMAGE_TOKEN_INDEX] = model.img_context_token_id

        keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1])
        logger.info(f'max_new_tokens: {max_new_tokens}')
        if max_new_tokens < 1:
            yield json.dumps({'text': ori_prompt + 'Exceeds max token length. Please start a new conversation, thanks.', 'error_code': 0}).encode() + b'\0'
            return

        thread = Thread(target=model.generate, kwargs=dict(
            input_ids=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=1.0,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            eos_token_id=eos_token_id,
            **image_args
        ))
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({'text': generated_text, 'error_code': 0}).encode() + b'\0'

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
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=21002)
    parser.add_argument('--worker-address', type=str,
        default='http://localhost:21002')
    parser.add_argument('--controller-address', type=str,
        default='http://localhost:21001')
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi-modal', action='store_true', help='Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.')
    parser.add_argument('--limit-model-concurrency', type=int, default=5)
    parser.add_argument('--stream-interval', type=int, default=1)
    parser.add_argument('--no-register', action='store_true')
    parser.add_argument('--load-8bit', action='store_true')
    parser.add_argument('--load-4bit', action='store_true')
    args = parser.parse_args()
    logger.info(f'args: {args}')

    if args.multi_modal:
        logger.warning('Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.')

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
