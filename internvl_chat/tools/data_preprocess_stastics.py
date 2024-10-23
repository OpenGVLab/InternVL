import json
from multiprocessing import Manager
import multiprocessing 
import argparse
from tqdm import tqdm
from functools import partial
import os
import numpy as np
from copy import deepcopy
import torch

from transformers import AutoTokenizer
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import find_closest_aspect_ratio
from internvl.conversation import get_conv_template
from torch.utils.data import Dataset
PROCESSES = 64


def get_num_patchs(orig_width, orig_height, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
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
    # target_width = image_size * target_aspect_ratio[0]
    # target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    if use_thumbnail and blocks != 1:
        blocks += 1
    return blocks


def preprocess_internlm(
    template_name,
    sources,
    tokenizer,
    num_image_token,
    text_only=False,
    group_by_length=False,
    num_image=1
):
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
            sentence['value'] = sentence['value'].strip()
            if sentence['value'][0] == '\n':
                sentence['value'] = sentence['value'][1:]
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    if not text_only:
        new_conversations = []
        for conversation in conversations:
            conversation = conversation.replace('<image>', image_tokens, num_image)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    return len(input_ids[0])


def preprocess_internlm_v2(
    template_name,
    sources,
    tokenizer,
    num_image_token,
    text_only=False,
    group_by_length=False,
    num_image=1
):
    num_image_token_list = [num_image_token]
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
            sentence['value'] = sentence['value'].strip()
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
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    return len(input_ids[0])


class DataProcess(Dataset):
    def __init__(self, template_name, meta, tokenizer, num_image_token, image_size=224, dynamic_image_size=False,
                 use_thumbnail=False, min_dynamic_patch=1, max_dynamic_patch=6, repeat_time=1, is_train=False,
                 pad2square=False, group_by_length=False, read_img=False, random_seed=0):
        super(DataProcess, self).__init__()
        self.template_name = template_name
        self.meta = meta
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token
        self.group_by_length = group_by_length
        self.image_size = image_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # choice top len(self.raw_data) * repeat_time samples
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        # for v2
        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)
    def __len__(self):
        return len(self.raw_data)

    def multi_modal_get_item(self, data_item):
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        orig_width, orig_height = data_item["width"], data_item["height"]
        num_patches = get_num_patchs(orig_width, orig_height, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                     image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # if not self.dynamic_image_size:
        #     assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        # if self.template_name == 'Hermes-2':
        #     preprocess_function = preprocess_mpt
        # elif self.template_name == 'internlm2-chat':
        # preprocess_function = preprocess_internlm
        preprocess_function = preprocess_internlm_v2
        # else:
        #     preprocess_function = preprocess
        num_tokens = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                         self.tokenizer, self.num_image_token * num_patches,
                                         group_by_length=True)

        ret = dict(
            num_patches=num_patches,
            num_tokens=num_tokens,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        num_patches = 1
        # preprocess_function = preprocess_internlm
        preprocess_function = preprocess_internlm_v2
        num_tokens = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                         self.tokenizer, self.num_image_token * num_patches,
                                         group_by_length=True)

        ret = dict(
            num_patches=num_patches,
            num_tokens=num_tokens,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def __getitem__(self, idx):
        idx = idx % len(self.raw_data)
        data_item = json.loads(self.raw_data[idx])
        if 'image' in data_item and data_item['image'] is not None and len(data_item['image']) != 0:
            ret = self.multi_modal_get_item(data_item)
        else:
            ret = self.pure_text_get_item(data_item)
        return ret


def decode_text(args):
    cfg_dataset, inds = args
    dataset = DataProcess(**cfg_dataset)
    dataset.ds_name = "dummy"
    token_lengths = []
    for idx in inds:
        item = dataset.__getitem__(idx)
        flag = item['image_flags'].sum().item()
        if flag == 0:
            num_vit_patch = item['num_patches']
            num_token = item['num_tokens']
            image_flags = 0
        elif flag == -1:
            num_vit_patch = -1
            num_token = -1
            image_flags = -1
        else:
            num_vit_patch = flag
            num_token = item['num_tokens']
            image_flags = flag

        token_lengths.append(
            {
                "vit_num": num_vit_patch,
                "token_num": num_token,
                "image_flags": image_flags
            }
        )

    return token_lengths


import copy
def worker(cfg_dataset, ds_name, token_lengths_path, ds_info):
    dataset = DataProcess(**cfg_dataset)
    with multiprocessing.Pool(PROCESSES) as pool:
        token_lengths_all = pool.map(decode_text, [(cfg_dataset, inds) for inds in np.array_split(range(len(dataset)), PROCESSES)])
    l_token_lengths = []
    for tmp in token_lengths_all:
        l_token_lengths.extend(tmp)

    length_save_path = os.path.join(token_lengths_path, f"{ds_name}"+"_token_lengths.json")

    with open(length_save_path, "w") as f:
        json.dump(l_token_lengths, f, indent=4)
    if "max_dynamic_patch" in ds_info:    
        info = {
            "root": ds_info["root"],
            "annotation": ds_info["annotation"],
            "data_augment": ds_info["data_augment"],
            "repeat_time": ds_info["repeat_time"],
            "length": len(dataset),
            "token_lengths": length_save_path,
            "max_dynamic_patch": ds_info["max_dynamic_patch"]
        }
    else:
        info = {
            "root": ds_info["root"],
            "annotation": ds_info["annotation"],
            "data_augment": ds_info["data_augment"],
            "repeat_time": ds_info["repeat_time"],
            "length": len(dataset),
            "token_lengths": length_save_path
        }
    return info


from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="data root path",
    )
    parser.add_argument(
        "--json_file",
        default=None,
        help="json file to statistics" 
    )
    parser.add_argument(
        "--worker",
        default=64, type=int,
        help="worker num",
    )
    parser.add_argument(
        "--token_lengths_path",
        default=None,
        help="token_lengths_path",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="token_lengths_path",
    )
    args = parser.parse_args()

    token_lengths_path = args.token_lengths_path

    # setting
    model_max_length = 4096
    tokenizer_path = "/path/to/tokenizer"
    data_path = args.json_file

    cfg_dataset_base = {
        'template_name': 'internlm2-chat',
        'num_image_token': 256,
        'image_size': 448,
        'dynamic_image_size': True,
        'use_thumbnail': True,
        'min_dynamic_patch': 1,
        'max_dynamic_patch': 4,
        'pad2square': False
    }

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = model_max_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)

    cfg_dataset_base['tokenizer'] = tokenizer
    
    ds_collections = json.loads(open(data_path).read())
    import time
    t_1 = time.time()
    meta = {}
    idx = 0
    for ds_name in tqdm(ds_collections.keys()):
        print(ds_name)
        cfg_dataset = copy.deepcopy(cfg_dataset_base)
        cfg_dataset['meta'] = ds_collections[ds_name]
        cfg_dataset['random_seed'] = idx
        ds_info = {}
        ds_info["root"] = ds_collections[ds_name]["root"]
        ds_info["annotation"] = ds_collections[ds_name]["annotation"]
        ds_info["data_augment"] = ds_collections[ds_name].get("data_augment", False)
        ds_info["repeat_time"] = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            ds_info['max_dynamic_patch'] = ds_collections[ds_name]['max_dynamic_patch']
        
        meta[ds_name] = worker(cfg_dataset, ds_name, token_lengths_path, ds_info)
        idx += 1

    with open(args.output_path, "w") as f:
        json.dump(meta.copy(), f, indent=4)

    t_2 = time.time()
    print(f"time: {t_2-t_1}")
