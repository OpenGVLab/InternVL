import argparse
import datetime
import io
import json
import os
import socket
import subprocess
import time
from collections import defaultdict

import torch
from lmdeploy import (GenerationConfig, TurbomindEngineConfig, VisionConfig,
                      pipeline)
from lmdeploy.vl.constants import IMAGE_TOKEN
from PIL import Image

try:
    from petrel_client.client import Client
    client = Client()
except:
    import warnings
    warnings.warn(
        'Fail to import petrel_client! '
        'You can ignore this warning if you do not need to load image from ceph.'
    )


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


IMG_PLACEHOLDER = '<image>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'


INSTRUCTION_EN = (
    'Your task is to answer the question below. '
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    '\n\n'
    'Question:'
    '\n\n'
    '{question}'
)

INSTRUCTION_ZH = (
    "你的任务是回答以下问题。在回答之前，请逐步推理说明您的思路。当你准备好给出答案时，请使用以下格式：\"答案: ...\""
    '\n\n'
    '问题:'
    '\n\n'
    '{question}'
)

VALID_INSTRUCTIONS = [
    'Answer the question using a single word or phrase.',
    "Answer with the option's letter from the given choices directly.",
    'Please answer Yes or No.',
]
VALID_INSTRUCTIONS = set(VALID_INSTRUCTIONS)


def init_distributed_mode():
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = rank % torch.cuda.device_count()

    world_size = int(os.environ['SLURM_NTASKS'])
    local_size = int(os.environ['SLURM_NTASKS_PER_NODE'])

    if 'MASTER_PORT' not in os.environ:
        port = 22222
        print(f'MASTER_PORT = {port}')
        os.environ['MASTER_PORT'] = str(port)

        time.sleep(3)

    node_list = os.environ['SLURM_NODELIST']
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr

    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(local_size)
    os.environ['WORLD_SIZE'] = str(world_size)

    torch.cuda.set_device(local_rank)


def localtime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def collate_fn(batches):
    items = []
    inputs = []
    for batch in batches:
        items.append(batch['item'])
        inputs.append((batch['question'], batch['image']))

    return inputs, items


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        sample_max_num=None,
    ):
        with open(data) as file:
            lines = file.readlines()

        self.data = []
        for line in lines:
            item = json.loads(line)

            multi_image = isinstance(item['image'], (list, tuple)) and len(item['image']) > 1

            if not (multi_image ^ args.multi_image):
                self.data.append(line)

        if sample_max_num is not None and len(self.data) > sample_max_num:
            print(f'Truncate data lines. {len(self.data)} => {sample_max_num}')
            step = len(self.data) // sample_max_num
            self.data = self.data[::step]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        question = item['question']
        images = item['image']
        if not isinstance(images, (list, tuple)):
            images = [images]

        images_new = []
        for image in images:
            if 's3://' in image:
                image = io.BytesIO(client.get(image))
            image = Image.open(image).convert('RGB')
            images_new.append(image)
        images = images_new

        for instruction in VALID_INSTRUCTIONS:
            if question.endswith(instruction):
                question = question[:-len(instruction)].strip()
        question = INSTRUCTION.format(question=question)

        if question.count(IMG_PLACEHOLDER) == 1:
            question = question.replace(IMG_PLACEHOLDER + '\n', '')
            question = question.replace(IMG_PLACEHOLDER, '')

        if question.count(IMG_PLACEHOLDER) == 0:
            question = IMG_PLACEHOLDER + '\n' + question

        return {
            'question': question.replace(IMG_PLACEHOLDER, IMAGE_TOKEN),
            'image': images,
            'item': item.copy(),
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def get_global_min(value):
    world_size = torch.distributed.get_world_size()
    merged_values = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_values, value)

    return min(merged_values)


def save_outputs(outputs, results_file):
    # multi_images = any(isinstance(x['image'], (list, tuple)) for x in outputs)
    # if multi_images:
    #     for x in outputs:
    #         x['image'] = list(x['image']) if isinstance(x['image'], (list, tuple)) else [x['image']]

    outputs = sorted(outputs, key=lambda x:x['image'])

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, outputs)

    merged_outputs = sum(merged_outputs, start=[])

    if torch.distributed.get_rank() == 0:
        with open(results_file, 'a') as file:
            for output in merged_outputs:
                file.write(json.dumps(output) + '\n')

        print(f'[{localtime()}] Results ({len(merged_outputs)=}) saved to {results_file}')


def evaluate_chat_model():
    dataset = VQADataset(
        data=args.prompt_path,
        sample_max_num=args.sample_max_num,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        sampler=InferenceSampler(len(dataset)),
    )
    min_len = get_global_min(len(dataloader))

    gen_config = GenerationConfig(
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
    )

    item2num = defaultdict(int)
    results_file = os.path.basename(args.prompt_path)
    results_file = os.path.join(args.out_dir, results_file)

    if args.multi_image:
        results_file = results_file.replace('.jsonl', '_ov.jsonl')

    if os.path.exists(results_file):
        with open(results_file) as file:
            lines = file.readlines()
        for line in lines:
            item = json.loads(line)
            item2num[(str(item['image']), item['question_orig'])] += 1

    print(
        f'[{localtime()}] [Rank {torch.distributed.get_rank()}] '
        f'Begin to answer {len(dataloader)} batches '
        f'(about {len(dataloader) * args.batch_size * args.num_return_sequences} samples), '
        f'{args.prompt_path=}, '
        f'{len(item2num)=}'
    )

    log_freq = max(len(dataloader) // args.batch_size // 100, 1)
    print_freq = max(len(dataloader) // args.batch_size // 100, 1)
    outputs = []
    for idx, (inputs, items) in enumerate(dataloader):
        assert len(inputs) == len(items)

        cnt_list = []
        filtered_items = []
        filtered_inputs = []
        for i in range(len(inputs)):
            cnt = args.num_return_sequences - item2num[(str(items[i]['image']), items[i]['question'])]
            if cnt <= 0:
                continue
            cnt_list.append(cnt)
            filtered_items.append(items[i])
            filtered_inputs.append(inputs[i])

        items = filtered_items
        inputs = filtered_inputs

        if len(inputs) <= 0:
            continue

        for _ in range(max(cnt_list)):
            gen_config.random_seed = None
            response_list = pipe(inputs, gen_config=gen_config)

            for input, item, response in zip(inputs, items, response_list):
                item = item.copy()
                item['question_orig'] = item['question']
                item['question'] = input[0].replace(IMAGE_TOKEN, IMG_PLACEHOLDER)
                item['response'] = response.text
                outputs.append(item)

        if idx % print_freq == 0 and torch.distributed.get_rank() == 0:
            print(
                f'[Prompt]\n{inputs[-1][0]}\n'
                f'[Image]\n{outputs[-1]["image"]}\n'
                f'[Question]\n{outputs[-1]["question_orig"]}\n'
                f'[Output]\n{outputs[-1]["response"]}\n'
                f'[Answer]\n{outputs[-1]["answer"]}\n'
                f'[End]\n'
            )

        if idx % log_freq == 0:
            print(
                f'[{localtime()}] '
                f'[Rank {torch.distributed.get_rank()}] '
                f'[Progress {idx}/{len(dataloader)}] '
            )

        if idx % log_freq == 0 and idx < min_len:
            save_outputs(outputs, results_file)
            outputs = []

    print(f'[{localtime()}] [Rank {torch.distributed.get_rank()}] Finish to generate')

    save_outputs(outputs, results_file)

    print(f'[{localtime()}] [Rank {torch.distributed.get_rank()}] Finish to save outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--prompt-path', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='sampled_outputs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--vit-batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--min-new-tokens', type=int, default=1)
    parser.add_argument('--num-return-sequences', type=int, default=8)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--multi-image', action='store_true')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--sample-max-num', type=int, default=None)
    parser.add_argument('--prompt-version', type=str, default='en', choices=['en', 'zh'])
    args = parser.parse_args()

    global INSTRUCTION
    if args.prompt_version == 'zh':
        INSTRUCTION = INSTRUCTION_ZH
    elif args.prompt_version == 'en':
        INSTRUCTION = INSTRUCTION_EN
    else:
        assert False, f'Unsupported prompt version {args.prompt_version}'

    assert args.num_return_sequences % args.batch_size == 0
    assert args.temperature > 0

    init_distributed_mode()

    model_name = '_'.join(args.checkpoint.split('/')[-2:])
    args.out_dir = os.path.join(args.out_dir, model_name, f'max_tiles_{args.max_num}')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    if int(os.getenv('RANK', '0')) % args.tp != 0:
        print(f"[SLURM_PROCID {int(os.environ['SLURM_PROCID'])}] Exit early")
        exit(0)

    if args.tp > 1:
        os.environ['RANK'] = str(int(os.environ['RANK']) // args.tp)
        os.environ['LOCAL_RANK'] = str(int(os.environ['LOCAL_RANK']) // args.tp)
        os.environ['WORLD_SIZE'] = str(int(os.environ['WORLD_SIZE']) // args.tp)
        # different rank should use different gpu, otherwise the all gather operation will be blocked
        torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=datetime.timedelta(days=10),
    )
    torch.distributed.barrier()
    print(f'world_size={torch.distributed.get_world_size()}, ip={socket.gethostbyname(socket.gethostname())}')

    vision_config = VisionConfig(max_batch_size=args.vit_batch_size)
    pipe = pipeline(
        args.checkpoint,
        vision_config=vision_config,
        backend_config=TurbomindEngineConfig(session_len=8192, cache_max_entry_count=0.1, tp=args.tp)
    )
    pipe.vl_encoder.model.config.max_dynamic_patch = args.max_num
    pipe.vl_encoder.model.config.dynamic_image_size = args.dynamic and not args.multi_image

    # lmdeploy will update the current_device
    torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    print(
        f'Begin to sample data from model {args.checkpoint}, '
        f'dynamic: {pipe.vl_encoder.model.config.dynamic_image_size}, '
        f'max_num: {pipe.vl_encoder.model.config.max_dynamic_patch}, '
    )
    evaluate_chat_model()
