# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
import json
import random
import logging
import warnings
import traceback

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import torch
import torch.distributed as dist
import transformers

from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    HfArgumentParser, Trainer, TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import enable_default_handler, enable_explicit_format, set_verbosity
from trl import DPOConfig

from internvl.dist_utils import init_dist
from internvl.model.internvl_chat import (
    InternVisionConfig, InternVisionModel,
    InternVLChatConfig, InternVLChatModel,
)
from internvl.patch import concat_pad_data_collator, dpo_concat_pad_data_collator, replace_gpt_oss_with_flash_sink_attn, replace_train_dataloader
from internvl.train.constants import (
    BOX_END_TOKEN, BOX_START_TOKEN,
    IMG_END_TOKEN, IMG_START_TOKEN,  IMG_CONTEXT_TOKEN,
    QUAD_END_TOKEN, QUAD_START_TOKEN,
    REF_END_TOKEN, REF_START_TOKEN,
)
from internvl.train.dataset import (
    ConcatDataset, TCSLoader,
    build_transform, dynamic_preprocess,
    preprocess_pretrain, preprocess_internvl2_5, preprocess_internvl3_5_gpt_oss,
)
from internvl.train.trainer_dpo import InternVLDPOTrainer
from internvl.model.internvl_chat.conversation import get_conv_template


USE_TCS_LOADER = bool(os.environ.get("USE_TCS_LOADER", "0") == "1")


R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()
R1_SYSTEM_PROMPT = R1_SYSTEM_PROMPT + "\n"


INSTRUCTION_BOXED_EN = (
    'Answer the preceding question. The last line of your response should follow this format: '
    "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your conclusion "
    'based on the reasoning provided. If you are uncertain or the problem is too complex, make '
    'a reasoned guess based on the information provided. Avoid repeating steps indefinitely—'
    'provide your best guess even if unsure. Think step by step logically, considering all '
    'relevant information before answering.'
).strip()


# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )
    use_custom_flash_attn: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the custom flash attn.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )
    split_annotations: bool = field(
        default=False,
        metadata={'help': 'Whether to split annotations to save memory usage. Default is False.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        split_annotations=False,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.split_annotations = split_annotations

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = random.sample(self.raw_data, k=int(len(self.raw_data) * repeat_time))
            if repeat_time > 1:
                repeat_time = int(repeat_time)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        if self.split_annotations:
            total_lines = len(self.raw_data)
            logger.info(f'world_size: {self.world_size}, rank: {self.rank}, total_lines: {total_lines}')
            lines_per_rank = total_lines // self.world_size  # Number of lines each rank should process
            lines_per_rank = max(1, lines_per_rank)

            # Calculate the start and end line numbers for the current rank
            start_line = lines_per_rank * self.rank  # Starting line for the current rank
            end_line = start_line + lines_per_rank  # Ending line for the current rank

            # Assign the appropriate lines to the current rank
            self.raw_data = self.raw_data[start_line:end_line]

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.num_fake_dump = 0

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self, use_pretrain=False):
        # Select the appropriate preprocessing function based on the template name
        if use_pretrain:
            return preprocess_pretrain

        if self.template_name == 'internvl2_5':
            return preprocess_internvl2_5

        if self.template_name == 'internvl3_5_gpt_oss':
            return preprocess_internvl3_5_gpt_oss

        raise NotImplementedError(f'Unsupported template: {self.template_name}')

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            return self.root + image_path
        return os.path.join(self.root, image_path)

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # if self.template_name == 'internvl3_5_gpt_oss':
        #     assert 'system_prompt' not in data_item

        #     conv = get_conv_template(self.template_name)
        #     system_prompt = conv.system_message

        #     if R1_SYSTEM_PROMPT in data_item['question']:
        #         data_item['question'] = data_item['question'].replace(R1_SYSTEM_PROMPT, '').strip()
        #         data_item['system_prompt'] = system_prompt
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Reasoning: low', 'Reasoning: high')
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Valid channels: final.', 'Valid channels: analysis, final.')

        #     elif INSTRUCTION_BOXED_EN in data_item['question']:
        #         data_item['question'] = data_item['question'].replace(INSTRUCTION_BOXED_EN, '').strip()
        #         data_item['system_prompt'] = system_prompt
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Reasoning: low', 'Reasoning: medium')

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['question']:
            data_item['question'] = '<image>\n' + data_item['question']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        chosen_conversations = [
            {'from': 'human', 'value': data_item['question']},
            {'from': 'gpt', 'value': data_item['chosen']},
        ]

        if 'system_prompt' in data_item:
            chosen_conversations.insert(0, {'from': 'system', 'value': data_item['system_prompt']})

        chosen_ret = preprocess_function(
            self.template_name,
            [deepcopy(chosen_conversations)],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=True,
            ds_name=self.ds_name,
        )

        rejected_conversations = [
            {'from': 'human', 'value': data_item['question']},
            {'from': 'gpt', 'value': data_item['rejected']},
        ]

        if 'system_prompt' in data_item:
            rejected_conversations.insert(0, {'from': 'system', 'value': data_item['system_prompt']})

        rejected_ret = preprocess_function(
            self.template_name,
            [deepcopy(rejected_conversations)],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=True,
            ds_name=self.ds_name,
        )

        prompt_mask = (chosen_ret['labels'][0] != IGNORE_INDEX).long()
        prompt_end_idx = prompt_mask.argmax().item() if prompt_mask.any() else -1

        assert (chosen_ret['input_ids'][0][:prompt_end_idx] == rejected_ret['input_ids'][0][:prompt_end_idx]).all()
        assert (chosen_ret['attention_mask'][0][:prompt_end_idx] == rejected_ret['attention_mask'][0][:prompt_end_idx]).all()

        # Create the final return dictionary
        ret = dict(
            prompt_input_ids=chosen_ret['input_ids'][0][:prompt_end_idx],
            prompt_attention_mask=chosen_ret['attention_mask'][0][:prompt_end_idx].long(),
            chosen_input_ids=chosen_ret['input_ids'][0][prompt_end_idx:],
            chosen_attention_mask=chosen_ret['attention_mask'][0][prompt_end_idx:].long(),
            rejected_input_ids=rejected_ret['input_ids'][0][prompt_end_idx:],
            rejected_attention_mask=rejected_ret['attention_mask'][0][prompt_end_idx:].long(),
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # if self.template_name == 'internvl3_5_gpt_oss':
        #     assert 'system_prompt' not in data_item

        #     conv = get_conv_template(self.template_name)
        #     system_prompt = conv.system_message

        #     if R1_SYSTEM_PROMPT in data_item['question']:
        #         data_item['question'] = data_item['question'].replace(R1_SYSTEM_PROMPT, '').strip()
        #         data_item['system_prompt'] = system_prompt
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Reasoning: low', 'Reasoning: high')
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Valid channels: final.', 'Valid channels: analysis, final.')

        #     elif INSTRUCTION_BOXED_EN in data_item['question']:
        #         data_item['question'] = data_item['question'].replace(INSTRUCTION_BOXED_EN, '').strip()
        #         data_item['system_prompt'] = system_prompt
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Reasoning: low', 'Reasoning: medium')

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]

        chosen_conversations = [
            {'from': 'human', 'value': data_item['question']},
            {'from': 'gpt', 'value': data_item['chosen']},
        ]

        if 'system_prompt' in data_item:
            chosen_conversations.insert(0, {'from': 'system', 'value': data_item['system_prompt']})

        chosen_ret = preprocess_function(
            self.template_name,
            [deepcopy(chosen_conversations)],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
            num_image=num_image,
        )

        rejected_conversations = [
            {'from': 'human', 'value': data_item['question']},
            {'from': 'gpt', 'value': data_item['rejected']},
        ]

        if 'system_prompt' in data_item:
            rejected_conversations.insert(0, {'from': 'system', 'value': data_item['system_prompt']})

        rejected_ret = preprocess_function(
            self.template_name,
            [deepcopy(rejected_conversations)],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
            num_image=num_image,
        )

        prompt_mask = (chosen_ret['labels'][0] != IGNORE_INDEX).long()
        prompt_end_idx = prompt_mask.argmax().item() if prompt_mask.any() else -1

        assert (chosen_ret['input_ids'][0][:prompt_end_idx] == rejected_ret['input_ids'][0][:prompt_end_idx]).all()
        assert (chosen_ret['attention_mask'][0][:prompt_end_idx] == rejected_ret['attention_mask'][0][:prompt_end_idx]).all()

        # Create the final return dictionary
        ret = dict(
            prompt_input_ids=chosen_ret['input_ids'][0][:prompt_end_idx],
            prompt_attention_mask=chosen_ret['attention_mask'][0][:prompt_end_idx].long(),
            chosen_input_ids=chosen_ret['input_ids'][0][prompt_end_idx:],
            chosen_attention_mask=chosen_ret['attention_mask'][0][prompt_end_idx:].long(),
            rejected_input_ids=rejected_ret['input_ids'][0][prompt_end_idx:],
            rejected_attention_mask=rejected_ret['attention_mask'][0][prompt_end_idx:].long(),
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # if self.template_name == 'internvl3_5_gpt_oss':
        #     assert 'system_prompt' not in data_item

        #     conv = get_conv_template(self.template_name)
        #     system_prompt = conv.system_message

        #     if R1_SYSTEM_PROMPT in data_item['question']:
        #         data_item['question'] = data_item['question'].replace(R1_SYSTEM_PROMPT, '').strip()
        #         data_item['system_prompt'] = system_prompt
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Reasoning: low', 'Reasoning: high')
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Valid channels: final.', 'Valid channels: analysis, final.')

        #     elif INSTRUCTION_BOXED_EN in data_item['question']:
        #         data_item['question'] = data_item['question'].replace(INSTRUCTION_BOXED_EN, '').strip()
        #         data_item['system_prompt'] = system_prompt
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Reasoning: low', 'Reasoning: medium')

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['question']:
            data_item['question'] = '<video>\n' + data_item['question']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['question'] = data_item['question'].replace('<video>\n', special_tokens)

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches

        chosen_conversations = [
            {'from': 'human', 'value': data_item['question']},
            {'from': 'gpt', 'value': data_item['chosen']},
        ]

        if 'system_prompt' in data_item:
            chosen_conversations.insert(0, {'from': 'system', 'value': data_item['system_prompt']})

        chosen_ret = preprocess_function(
            self.template_name,
            [deepcopy(chosen_conversations)],
            self.tokenizer,
            num_image_tokens,
            group_by_length=True,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_patches,
        )

        rejected_conversations = [
            {'from': 'human', 'value': data_item['question']},
            {'from': 'gpt', 'value': data_item['rejected']},
        ]

        if 'system_prompt' in data_item:
            rejected_conversations.insert(0, {'from': 'system', 'value': data_item['system_prompt']})

        rejected_ret = preprocess_function(
            self.template_name,
            [deepcopy(rejected_conversations)],
            self.tokenizer,
            num_image_tokens,
            group_by_length=True,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_patches,
        )

        prompt_mask = (chosen_ret['labels'][0] != IGNORE_INDEX).long
        prompt_end_idx = prompt_mask.argmax().item() if prompt_mask.any() else -1

        assert (chosen_ret['input_ids'][0][:prompt_end_idx] == rejected_ret['input_ids'][0][:prompt_end_idx]).all()
        assert (chosen_ret['attention_mask'][0][:prompt_end_idx] == rejected_ret['attention_mask'][0][:prompt_end_idx]).all()

        # Create the final return dictionary
        ret = dict(
            prompt_input_ids=chosen_ret['input_ids'][0][:prompt_end_idx],
            prompt_attention_mask=chosen_ret['attention_mask'][0][:prompt_end_idx].long(),
            chosen_input_ids=chosen_ret['input_ids'][0][prompt_end_idx:],
            chosen_attention_mask=chosen_ret['attention_mask'][0][prompt_end_idx:].long(),
            rejected_input_ids=rejected_ret['input_ids'][0][prompt_end_idx:],
            rejected_attention_mask=rejected_ret['attention_mask'][0][prompt_end_idx:].long(),
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # if self.template_name == 'internvl3_5_gpt_oss':
        #     assert 'system_prompt' not in data_item

        #     conv = get_conv_template(self.template_name)
        #     system_prompt = conv.system_message

        #     if R1_SYSTEM_PROMPT in data_item['question']:
        #         data_item['question'] = data_item['question'].replace(R1_SYSTEM_PROMPT, '').strip()
        #         data_item['system_prompt'] = system_prompt
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Reasoning: low', 'Reasoning: high')
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Valid channels: final.', 'Valid channels: analysis, final.')

        #     elif INSTRUCTION_BOXED_EN in data_item['question']:
        #         data_item['question'] = data_item['question'].replace(INSTRUCTION_BOXED_EN, '').strip()
        #         data_item['system_prompt'] = system_prompt
        #         data_item['system_prompt'] = data_item['system_prompt'].replace('Reasoning: low', 'Reasoning: medium')

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        chosen_conversations = [
            {'from': 'human', 'value': data_item['question']},
            {'from': 'gpt', 'value': data_item['chosen']},
        ]

        if 'system_prompt' in data_item:
            chosen_conversations.insert(0, {'from': 'system', 'value': data_item['system_prompt']})

        chosen_ret = preprocess_function(
            self.template_name,
            [deepcopy(chosen_conversations)],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=True,
            ds_name=self.ds_name,
        )

        rejected_conversations = [
            {'from': 'human', 'value': data_item['question']},
            {'from': 'gpt', 'value': data_item['rejected']},
        ]

        if 'system_prompt' in data_item:
            rejected_conversations.insert(0, {'from': 'system', 'value': data_item['system_prompt']})

        rejected_ret = preprocess_function(
            self.template_name,
            [deepcopy(rejected_conversations)],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=True,
            ds_name=self.ds_name,
        )

        prompt_mask = (chosen_ret['labels'][0] != IGNORE_INDEX).long()
        prompt_end_idx = prompt_mask.argmax().item() if prompt_mask.any() else -1

        assert (chosen_ret['input_ids'][0][:prompt_end_idx] == rejected_ret['input_ids'][0][:prompt_end_idx]).all()
        assert (chosen_ret['attention_mask'][0][:prompt_end_idx] == rejected_ret['attention_mask'][0][:prompt_end_idx]).all()

        # Create the final return dictionary
        ret = dict(
            prompt_input_ids=chosen_ret['input_ids'][0][:prompt_end_idx],
            prompt_attention_mask=chosen_ret['attention_mask'][0][:prompt_end_idx].long(),
            chosen_input_ids=chosen_ret['input_ids'][0][prompt_end_idx:],
            chosen_attention_mask=chosen_ret['attention_mask'][0][prompt_end_idx:].long(),
            rejected_input_ids=rejected_ret['input_ids'][0][prompt_end_idx:],
            rejected_attention_mask=rejected_ret['attention_mask'][0][prompt_end_idx:].long(),
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
        )
        return ret

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                # raise StopIteration
                return self.fake_data_get_item()
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                # print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
    split_annotations=False,
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name].get('data_augment', False),
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            split_annotations=split_annotations,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        lengths.append(len(dataset))

    train_dataset = ConcatDataset(datasets)
    return train_dataset


def main():
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DPOConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.loss_type = ['sigmoid', 'bco_pair', 'sft']
    training_args.loss_weights = [0.8, 0.2, 1.0]
    training_args.remove_unused_columns = False
    training_args.max_length = data_args.max_seq_length
    training_args.gradient_checkpointing = model_args.grad_checkpoint

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('petreloss.conf') if USE_TCS_LOADER else None

    if model_args.use_liger:
        raise NotImplementedError

    logger.info('Loading InternVLChatModel...')
    config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_config.drop_path_rate = model_args.drop_path_rate

    if config.llm_config.model_type == 'gpt_oss':
        config.llm_config.output_router_logits = True

        config.output_router_logits = config.llm_config.output_router_logits
        config.router_aux_loss_coef = config.llm_config.router_aux_loss_coef

        logger.info('Using vanilla attention for GptOssForCausalLM')

        # config.llm_config._attn_implementation = 'kernels-community/vllm-flash-attn3'
        # logger.info('Using kernels-community/vllm-flash-attn3 for GptOssForCausalLM')
    else:
        config.llm_config._attn_implementation = 'flash_attention_2'
        logger.info('Using flash_attention_2')

    config.template = data_args.conv_style
    config.select_layer = model_args.vision_select_layer
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail = data_args.use_thumbnail
    config.ps_version = model_args.ps_version
    config.min_dynamic_patch = data_args.min_dynamic_patch
    config.max_dynamic_patch = data_args.max_dynamic_patch
    model = InternVLChatModel.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    ref_model = InternVLChatModel.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)

    model.img_context_token_id = img_context_token_id
    ref_model.img_context_token_id = img_context_token_id
    model.tokenizer = tokenizer

    if model_args.use_custom_flash_attn:
        replace_gpt_oss_with_flash_sink_attn(model.language_model)
        replace_gpt_oss_with_flash_sink_attn(ref_model.language_model)

    assert model.config.downsample_ratio == data_args.down_sample_ratio
    assert ref_model.config.downsample_ratio == data_args.down_sample_ratio

    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    ref_model.config.force_image_size = data_args.force_image_size
    ref_model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    if model_args.grad_checkpoint:
        model.vision_model.gradient_checkpointing = True
        model.vision_model.encoder.gradient_checkpointing = True
        model.language_model._set_gradient_checkpointing()
        logger.info('gradient_checkpointing is enabled')

    train_dataset = build_datasets(
        data_args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
        split_annotations=data_args.split_annotations,
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    training_args.split_annotations = data_args.split_annotations
    trainer = InternVLDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=dpo_concat_pad_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        print(f'[Memory Usage before training] {torch.cuda.memory_allocated()/1024/1024/1024:.2f}GB')
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
