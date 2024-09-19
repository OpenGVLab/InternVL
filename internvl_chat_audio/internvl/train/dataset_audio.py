import sys
import logging
import os
import io
import re
import json
import random
import copy
import math
from copy import deepcopy
import traceback
from PIL import PngImagePlugin, Image, ImageFile
import imageio
from transformers.trainer_pt_utils import LabelSmoother
import librosa
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from typing import Dict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import transformers
import decord
from decord import VideoReader
from decord import cpu
from internvl.conversation import get_conv_template
from PIL import Image
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist
from internvl.train.dataset_interleaved_iterable import (PackedDataset,
                                                         packed_collate_fn)
from torchvision import transforms
from .dataset import LazySupervisedDataset, dynamic_preprocess
from .constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD,
                        AUDIO_START_TOKEN, AUDIO_END_TOKEN, AUDIO_CONTEXT_TOKEN)

try:
    # from aoss_client.client import Client
    from petrel_client.client import Client
    # from petrel_client.common.config import Config
except ImportError as E:
    print('please install petrel_client')

logger = logging.getLogger(__name__)

decord.bridge.set_bridge("torch")

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
ImageFile.LOAD_TRUNCATED_IMAGES = True

class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')


class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn):
        if "s3://" in fn:
            img_value_str = self.client.get(fn)
            img = pil_loader(img_value_str)
        else:
            # load from local (or s3mount node)
            img = Image.open(fn).convert("RGB")
        return img

class TCSAudioLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSAudioLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn, sr=16000):
        try:
            if  "s3://" in fn: #fn.startswith("s3://"):
                audio_value_str = self.client.get(fn)
                if len(audio_value_str) < 1000:
                    return None
                audio, _ = librosa.load(io.BytesIO(audio_value_str), sr=sr)
            else:
                audio, _ = librosa.load(fn, sr=sr)
            return audio
        except:
            return None

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_gif(
        video_path, num_frames, sample='rand', fix_start=None,
         max_num_frames=-1, client=None
    ):
    if  "s3://" in video_path:#  video_path.startswith('s3'):
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        max_num_frames=max_num_frames
    )
    frames = []
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    frames = torch.stack(frames)  # .float() / 255
    return frames, frame_indices, None


def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None,
        max_num_frames=-1, client=None, clip=None
    ):
    if  "s3://" in video_path: # video_path.startswith('s3'):
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, float(fps)


def read_frames(
        video_path, num_frames, sample='rand', fix_start=None,
        max_num_frames=-1, client=None, clip=None
    ):
    if  "s3://" in video_path: #video_path.startswith('s3://'):
        image_list = client.list(video_path)
        image_list = sorted(image_list)
        frames = [Image.open(io.BytesIO(client.get(fn))) for fn in image_list]
    else:
        image_list = sorted(list(os.listdir(video_path)))
        frames = []
        for image in image_list:
            fp = os.path.join(video_path, image)
            frame = Image.open(fp)
            frames.append(frame)

    vlen = len(frames)
    if vlen > num_frames:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start, max_num_frames=max_num_frames
            )
        frames = [frames[i] for i in frame_indices]
    return frames

class TCSVideoLoader(object):
    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSAudioLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn, num_frames, sample='rand'):
        try:
            self.video_transform = transforms.Lambda(lambda x: x.float().div(255.0))
            if fn.endswith('/'):
                frames = read_frames(fn, num_frames, client=self.client, sample=sample)
            elif fn.endswith('.gif'):
                frames, frame_indices, fps = read_frames_gif(fn, num_frames, client=self.client, sample=sample)
                frames = self.video_transform(frames)
            else:
                frames, frame_indices, fps = read_frames_decord(fn, num_frames, client=self.client, sample=sample)
                frames = self.video_transform(frames)
            return frames
        except:
            return None


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform

def check_audio_conversations(conversations, num_audio_token):
    audio_cnt = 0
    for idx, conv in enumerate(conversations):
        audio_cnt += conv['value'].count('<audio>')
    assert audio_cnt == len(num_audio_token), f'There should be {num_audio_token} in the conversation, but got {audio_cnt}'

    return conversations


def build_video_transform(input_size, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD

    video_transform = transforms.Compose(
            [
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.Normalize(MEAN, STD)
            ]
        )

    return video_transform

def preprocess(template_name,
               sources,
               tokenizer: transformers.PreTrainedTokenizer,
               num_image_token_list: list,
               text_only: bool = False,
               group_by_length: bool = False,
               use_packed_ds: bool = False,
               ds_name: str = None,
               num_image: int = 1
               ) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    # image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            logger.info(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_mpt(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    # image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|im_end|><|im_start|>assistant\n
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            turn_len = len(tokenizer(turn).input_ids) + 1

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids)

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_phi3(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    # image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    tokenizer.padding_side = 'right'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|end|>\n<|assistant|>
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(int(tokenizer.pad_token_id)).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        target[target == endoftext_id] = IGNORE_TOKEN_ID

        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            if i == 0:
                turn_len = len(tokenizer(turn).input_ids)
            else:
                turn_len = len(tokenizer(turn).input_ids) - 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if i == 0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            else:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess_llama3(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())


    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    tokenizer.padding_side = 'right'
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1]  # <|end|>\n<|assistant|>
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep)
        re_turns = [conv.sep.join(turns[:3])]  # system + user + gpt
        for conv_idx in range(3, len(turns), 2):
            re_turns.append(conv.sep.join(turns[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(re_turns):
            if turn == '':
                break
            if i == 0:
                turn_len = len(tokenizer(turn).input_ids)
            else:
                turn_len = len(tokenizer(turn).input_ids) + 1
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if i == 0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            else:
                instruction_len = len(tokenizer(parts[0]).input_ids)

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
            # print(f'[question {i}]', tokenizer.decode(input_ids[:, cur_len: cur_len + instruction_len][0]))
            # print(f'[answer {i}]', tokenizer.decode(input_ids[:, cur_len + instruction_len: cur_len + turn_len][0]))
            # print(f'[label {i}]', target[cur_len + instruction_len: cur_len + turn_len])
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, int(tokenizer.pad_token_id), z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}.'
                    f' #turn = {len(turns) - 1}. (ignored). This dataset is {ds_name}.'
                )
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

##internlm

def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        video_frame_token: int,
        num_video_frames: int,
        num_audio_token: torch.LongTensor,
        text_only: bool = False,
        truncation: bool = True,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1,
        mask_label: bool = True,
) -> Dict:
    conv = get_conv_template(template_name)
    roles =  {"system": conv.roles[0], "human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        if num_audio_token.max() > 0:
            source = check_audio_conversations(source, num_audio_token)

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            # assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].lstrip()  # 去除左边的空白字符，右边的空白字符比如<audio>\n保留
            if sentence['value'][0] == '\n':
                sentence['value'] = sentence['value'][1:]
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())


    video_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * video_frame_token}{IMG_END_TOKEN}'* num_video_frames
    audio_tokens_generator = lambda num_audio_token: f"{AUDIO_START_TOKEN}{AUDIO_CONTEXT_TOKEN * num_audio_token}{AUDIO_END_TOKEN}"

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    def replace_audio_token(match):
        count = int(num_audio_token[replace_audio_token.counter].item())
        replace_audio_token.counter += 1
        return audio_tokens_generator(count)

    replace_audio_token.counter = 0

    if num_audio_token.max() > 0:
        if int(torch.max(num_audio_token).item()) > 0:
            new_conversations = []
            for conversation in conversations:
                conversation = re.sub(r"<audio>", replace_audio_token, conversation) # 逐一替换成对应长度的token
                new_conversations.append(conversation)
            conversations = new_conversations

    if num_video_frames > 0:
        new_conversations = []
        for conversation in conversations:
            conversation = conversation.replace('<video>', video_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=truncation,
    ).input_ids
    targets = input_ids.clone()
    if not mask_label:
        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )

    for conversation, target in zip(conversations, targets):
        assert target[-1].item() != tokenizer(AUDIO_CONTEXT_TOKEN).input_ids[-1], f'audio token too long'
        assert torch.sum(target == tokenizer(AUDIO_CONTEXT_TOKEN).input_ids[-1]).item() == torch.sum(num_audio_token).item(), f'audio token is truncated'
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.')
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


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
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def find_minimal_patch_aspect_ratio(target_ratios, orig_width, orig_height, image_size, scale_threshold=1.0):
    max_gain = float('-inf')
    for ratio in target_ratios:
        scale_factor = min(
            ratio[0] * image_size / orig_width,
            ratio[1] * image_size / orig_height,
        )
        gain = min(scale_factor, scale_threshold)
        if gain > max_gain:
            max_gain = gain
            best_scale_factor = scale_factor
            best_ratio = ratio

    return best_ratio, best_scale_factor




class LazySupervisedAudioDataset(LazySupervisedDataset):
    default_seed = 42
    """Dataset for supervised fine-tuning."""

    def __init__(self, template_name, meta, tokenizer, audio_processor, tcs_loader, tcs_audio_loader, tcs_video_loader, sampling_method, video_max_frame_num,
                 ds_name, speech_aug_processor, speech_augment_ratio, num_image_token,
                 image_size=224, is_train=True, pad2square=False, group_by_length=False, padding_max_length=False,
                 dynamic_image_size=False, use_thumbnail=False, min_dynamic_patch=1,
                 max_dynamic_patch=6, repeat_time=1, normalize_type='imagenet', scale_threshold="old",
                 # hyperparameters for packed training
                use_packed_ds=False,
                data_rank=0,
                data_world_size=1,
                distributed_mode=False,
                force_shuffle=False,
                random_seed=0,):
        super().__init__(template_name, meta, tokenizer, tcs_loader, ds_name, num_image_token,
                 image_size, is_train, pad2square, group_by_length,
                 dynamic_image_size, use_thumbnail, min_dynamic_patch,
                 max_dynamic_patch, repeat_time, normalize_type,
                 use_packed_ds, data_rank, data_world_size, distributed_mode, force_shuffle, random_seed)

        self.tcs_audio_loader = tcs_audio_loader
        self.tcs_video_loader = tcs_video_loader
        self.video_max_frame_num = video_max_frame_num
        self.sampling_method = sampling_method
        self.audio_path = meta.pop('audio_path', None)  # root audio path
        self.image_path = meta.pop('image_path', None)  # root image path
        self.video_path = meta.pop('video_path', None)
        self.audio_processor = audio_processor
        self.padding_max_length = padding_max_length

        if self.group_by_length or self.use_packed_ds :
            self.padding = False
        elif self.padding_max_length:
            self.padding = 'max_length'
        else:
            self.padding = False
        self.speech_aug_processor = speech_aug_processor
        self.speech_augment_ratio = speech_augment_ratio
        self.video_transform = build_video_transform(input_size=self.image_size, normalize_type=self.normalize_type)

    def __len__(self):
        return len(self.raw_data)

    def multi_modal_get_item(self, data_item):
        # 检查图片输入，只有第一轮对话的问题中才能出现<image>
        has_image = 'image' in data_item and data_item['image'] is not None and len(data_item['image']) != 0
        if has_image:
            image_cnt = 0
            for idx, conv in enumerate(data_item['conversations']):
                conv['value'] = conv['value'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
                if idx == 0:
                    conv['value'] = '<image>\n' + conv['value']
                image_cnt += conv['value'].count('<image>')
            assert image_cnt == 1, f'There should be exactly one <image> in the conversation, but got {image_cnt}'

        # 检查音频输入
        has_audio = 'audio' in data_item and data_item['audio'] is not None and len(data_item['audio']) != 0  # 确定是否与音频输入
        if has_audio:
            audio_cnt = 0
            audio_list = data_item.pop('audio', '')
            if not isinstance(audio_list, list):
                audio_list = [audio_list, ]
            for idx, conv in enumerate(data_item['conversations']):
                if idx % 2 == 0:
                    audio_cnt += conv['value'].count('<audio>')
                else: # 答案中不能出现<audio>
                    conv['value'] = conv['value'].replace('<audio>\n', '').replace('\n<audio>', '').replace('<audio>', '')

            assert audio_cnt == len(audio_list), f'There should be exactly {len(audio_list)} <audio> in the conversation, but got {audio_cnt}'

        # 检测video输入
        has_video = 'video' in data_item and data_item['video'] is not None and len(data_item['video']) != 0  # 确定是否有视频输入
        if has_video:
            video_cnt = 0
            for idx, conv in enumerate(data_item['conversations']):
                conv['value'] = conv['value'].replace('<video>\n', '').replace('\n<video>', '').replace('<video>', '')
                if idx == 0:
                    conv['value'] = '<video>\n' + conv['value']
                video_cnt += conv['value'].count('<video>')

            assert video_cnt == 1, f'There should be exactly one <video> in the conversation, but got {video_cnt}'


        if has_image:
            if data_item['image'].startswith('s3://'):
                img_path = os.path.join(self.image_path, data_item['image'])
            else:
                img_path = os.path.join(self.image_path, data_item['image'])
            if self.tcs_loader is not None:
                image = self.tcs_loader(img_path)
            else:
                image = Image.open(img_path).convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), (255, 255, 255))

        if has_video:
            vid_file = os.path.join(self.video_path, data_item['video'])
            video_values = self.tcs_video_loader(vid_file, self.video_max_frame_num, self.sampling_method)
            assert video_values is not None and len(video_values) != 0, f'video values of {vid_file} is None.'
        else:
            video_values = [Image.new('RGB', (224, 224), (255, 255, 255)) for _ in range(2)]
        #process audio
        if not has_audio:
            audio_values = np.zeros((480000,), dtype=np.float32)
            audio_values = self.audio_processor(audio_values, sampling_rate=16000, return_tensors="pt")
            input_features = [audio_values['input_features']]
            audio_len_after_cnn = [audio_values['audio_len_after_cnn']]
            audio_token_num = [audio_values['audio_token_num']]
            num_audio = 1

        else:
            if not isinstance(audio_list, list):
                audio_list = [audio_list, ]
            num_audio = len(audio_list)
            input_features = []
            audio_len_after_cnn = []
            audio_token_num = []
            for audio_file in audio_list:
                audio_file = os.path.join(self.audio_path, audio_file)
                audio_values = self.tcs_audio_loader(audio_file, sr=16000) # sample rate should be 16000
                assert audio_values is not None, f'audio values of {audio_file} is None.'
                if self.speech_aug_processor is not None:
                    if random.random() < self.speech_augment_ratio:
                        audio_values = self.speech_aug_processor.aug(audio_values)
                audio_values = self.audio_processor(audio_values, sampling_rate=16000, return_tensors="pt")
                # print('audio_values['input_features'].shape:', audio_values['input_features'].shape)  [1, 128, 3000]
                input_features.append(audio_values['input_features'])
                audio_len_after_cnn.append(audio_values['audio_len_after_cnn'])
                audio_token_num.append(audio_values['audio_token_num'])
                assert audio_values['audio_token_num'].item() != 0, f'audio_token_num of {audio_file} is 0.'

        input_features = torch.cat(input_features, dim=0)  # [num_audio, 128, 3000]
        audio_len_after_cnn = torch.stack(audio_len_after_cnn, dim=0)  # [num_audio]
        audio_token_num = torch.stack(audio_token_num, dim=0)  # [num_audio]

        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        if self.dynamic_image_size:

            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        if isinstance(video_values, list):
            video_values = [transform(d) for d in video_values] # load the floders that contain frames
            video_values = torch.stack(video_values, dim=0)
        else:
            video_values = self.video_transform(video_values)  # load the raw videos
        video_num_patches = video_values.size(0)
        assert video_num_patches <= self.video_max_frame_num, f'video frames are too long'

        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        else:
            preprocess_function = preprocess

        visual_values = pixel_values
        visual_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        if has_image and has_video:
            visual_values = torch.cat([pixel_values, video_values], dim=0)
            visual_flags=torch.tensor([1] * (num_patches + video_num_patches), dtype=torch.long)
        elif has_image:
            visual_values = pixel_values
            visual_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        elif has_video:
            visual_values = video_values
            visual_flags=torch.tensor([1] * video_num_patches, dtype=torch.long)

        ret = preprocess_function(
            self.template_name, 
            [deepcopy(data_item['conversations'])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            self.num_image_token,
            0 if has_video == False else video_num_patches,
            torch.tensor([0]) if has_audio == False else audio_token_num,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name)

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=visual_values,
            image_flags=visual_flags,
            audio_values=input_features,
            audio_len_after_cnn=audio_len_after_cnn,
            audio_token_num=audio_token_num,
            audio_flags= torch.tensor([0] * num_audio, dtype=torch.long) if has_audio == False else torch.tensor([1] * num_audio, dtype=torch.long)
        )
        return ret


    def pure_text_get_item(self, data_item):
        

        # fake image
        image = Image.new('RGB', (224, 224), (255, 255, 255))
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # fake audio
        audio_values = np.zeros((480000,), dtype=np.float32)
        audio_values = self.audio_processor(audio_values, sampling_rate=16000, return_tensors="pt")
        input_features = [audio_values['input_features']]
        audio_len_after_cnn = [audio_values['audio_len_after_cnn']]
        audio_token_num = [audio_values['audio_token_num']]
        input_features = torch.cat(input_features, dim=0)
        audio_len_after_cnn = torch.stack(audio_len_after_cnn, dim=0)
        audio_token_num = torch.stack(audio_token_num, dim=0)
        num_audio = 1

        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        else:
            preprocess_function = preprocess
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, self.num_image_token * num_patches, self.num_image_token, 0, torch.tensor([0]), text_only=True, ds_name=self.ds_name,
                                  padding=self.padding, mask_label=True, truncation=True)
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            audio_values=input_features,
            audio_len_after_cnn=audio_len_after_cnn,
            audio_token_num=audio_token_num,
            audio_flags= torch.tensor([0] * num_audio, dtype=torch.long)
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        i = i % len(self.raw_data)

        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                has_image = 'image' in data_item and data_item['image'] is not None and len(data_item['image']) != 0
                has_audio = 'audio' in data_item and data_item['audio'] is not None and len(data_item['audio']) != 0
                has_video = 'video' in data_item and data_item['video'] is not None and len(data_item['video']) != 0
                multi_modal_flag = has_image or has_audio or has_video
                if multi_modal_flag:
                    ret = self.multi_modal_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)  # 纯语言数据 text=>text
                break
            except Exception as e:
                print(e)
                data_item = json.loads(self.raw_data[i])
                if has_image and self.image_path:
                    data_path = os.path.join(self.image_path, data_item['image'])
                    print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}, data_item={data_item}, msg={traceback.format_exc()}')
                elif has_audio and self.audio_path:
                    print(f'Failed to load audio, the dataset is: {self.ds_name}, data_item={data_item}, msg={traceback.format_exc()}')
                elif has_video and self.video_path:
                    print(f'Failed to load video, the dataset is: {self.ds_name}, data_item={data_item}, msg={traceback.format_exc()}')
                i = random.randint(0, len(self.raw_data) - 1)
                return self.__getitem__(i)
        return ret




def build_datasets(
        data_args,
        tokenizer,
        audio_processor,
        tcs_loader,
        tcs_audio_loader,
        tcs_video_loader,
        model,
        sampling_method,
        video_max_frame_num=8,
        speech_aug_processor=None,
        speech_augment_ratio=0.5,
        group_by_length=False,
        padding_max_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        normalize_type='imagenet'):
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(
                f'max_dynamic_patch is set to {max_num} according to the meta file'
            )
        else:
            max_num = max_dynamic_patch
        is_train = data_args.force_image_aug or ds_collections[ds_name]['data_augment']
    
        dataset = LazySupervisedAudioDataset(
            data_args.conv_style,
            ds_collections[ds_name],
            tokenizer,
            audio_processor,
            tcs_loader,
            tcs_audio_loader,
            tcs_video_loader,
            sampling_method,
            video_max_frame_num,
            ds_name=ds_name,
            speech_aug_processor=speech_aug_processor,
            speech_augment_ratio=speech_augment_ratio,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=is_train,
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            padding_max_length=padding_max_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        

        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)

    return train_dataset


def get_token_sum(g):
    sum = 0
    for i in g:
        sum += i[2]
    return sum

def get_vit_num(g):
    vit_num = 0
    for _ in g:
        vit_num += _[1]
    return vit_num
