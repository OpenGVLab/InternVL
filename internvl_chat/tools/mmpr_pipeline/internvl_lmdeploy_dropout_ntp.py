import argparse
import datetime
import io
import json
import os
import random
import socket
import subprocess
import time
from collections import defaultdict

import torch
from lmdeploy import TurbomindEngineConfig, VisionConfig, pipeline
from lmdeploy.model import InternVL2InternLM2, Qwen7BChat
from lmdeploy.vl.constants import IMAGE_TOKEN
from petrel_client.client import Client
from PIL import Image

client = Client()
IMG_PLACEHOLDER = '<image>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'


def messages2prompt(self, messages, sequence_start=True, **kwargs):
    """Return the prompt that is concatenated with other elements in the
    chat template.

    Args:
        messages (str | List): user's input prompt
    Returns:
        str: the concatenated prompt
    """
    if isinstance(messages, str):
        return self.get_prompt(messages, sequence_start)

    prefix_info = None
    if messages[-1]['role'] == 'prefix':
        prefix_info = messages.pop(-1)
        prefix_info = prefix_info['content']

    box_map = dict(user=self.user,
                    assistant=self.assistant,
                    system=self.system)
    eox_map = dict(user=self.eoh,
                    assistant=self.eoa + self.separator,
                    system=self.eosys)
    ret = ''
    if self.meta_instruction is not None and sequence_start:
        if len(messages) and messages[0]['role'] != 'system':
            ret += f'{self.system}{self.meta_instruction}{self.eosys}'
    for message in messages:
        role = message['role']
        content = message['content']
        ret += f'{box_map[role]}{content}{eox_map[role]}'
    ret += f'{self.assistant}'

    if prefix_info is not None:
        ret += f'{prefix_info}'

    return ret


def init_distributed_mode():
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = rank % torch.cuda.device_count()

    world_size = int(os.environ['SLURM_NTASKS'])
    local_size = int(os.environ['SLURM_NTASKS_PER_NODE'])

    if 'MASTER_PORT' not in os.environ:
        port = 22222
        # for i in range(22222, 65535):
        #     cmd = f'netstat -aon|grep {i}'
        #     with os.popen(cmd, 'r') as file:
        #         if '' == file.read():
        #             port = i
        #             break

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
    prefixs = []
    for batch in batches:
        items.append(batch['item'])
        if 'image' in batch:
            inputs.append((batch['question'], batch['image']))
        else:
            inputs.append(batch['question'])
        prefixs.append(batch['prefix'])

    return inputs, prefixs, items


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        start_ratio=0.5,
        sample_max_num=None,
        load_image=False,
    ):
        with open(data) as file:
            self.data = file.readlines()

        self.start_ratio = start_ratio
        self.load_image = load_image

        if sample_max_num is not None and len(self.data) > sample_max_num:
            print(f'Truncate data lines. {len(self.data)} => {sample_max_num}')
            step = len(self.data) // sample_max_num
            self.data = self.data[::step]

    def __len__(self):
        return len(self.data)

    def _truncate_prefix(self, prefix):
        splitted_prefix = prefix.split(' ')
        sep_idx = int(len(splitted_prefix) * self.start_ratio)
        splitted_prefix = splitted_prefix[:sep_idx]
        return ' '.join(splitted_prefix).strip()

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        question = item['question']
        prefix = self._truncate_prefix(item['chosen'])

        if self.load_image:
            images = item['image']
            if not isinstance(images, (list, tuple)):
                images = [images]

            images_new = []
            for image in images:
                ip_idx = random.randint(1, 3)
                image = image.replace('wenhaitmp:s3://internvl/', 'langchao:s3://internvl2/')
                image = image.replace('langchao:s3://', f'langchao_ip{ip_idx}:s3://')

                if 's3://' in image:
                    image = io.BytesIO(client.get(image))
                image = Image.open(image).convert('RGB')
                images_new.append(image)

            images = images_new

            return {
                'question': question.replace(IMG_PLACEHOLDER, IMAGE_TOKEN),
                'image': images,
                'prefix': prefix,
                'item': item.copy(),
            }

        return {
            'question': question.replace(IMG_PLACEHOLDER, IMAGE_TOKEN),
            'prefix': prefix,
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


def evaluate_chat_model():
    dataset = VQADataset(
        data=args.prompt_path,
        start_ratio=args.start_ratio,
        sample_max_num=args.sample_max_num,
        load_image=args.load_image,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        sampler=InferenceSampler(len(dataset)),
    )

    generation_config = dict(
        request_output_len=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    item2num = defaultdict(int)
    results_file = os.path.basename(args.prompt_path)
    results_file = os.path.join(args.out_dir, results_file)
    if os.path.exists(results_file):
        with open(results_file) as file:
            lines = file.readlines()
        for line in lines:
            item = json.loads(line)
            item2num[(str(item['image']), item['question'])] += 1

    print(
        f'[Rank {torch.distributed.get_rank()}] '
        f'Begin to answer {len(dataloader)} batches '
        f'(about {len(dataloader) * args.batch_size} samples), '
        f'{args.prompt_path=}, '
        f'{len(item2num)=}'
    )

    log_freq = max(len(dataloader) // 100, 1)
    print_freq = max(len(dataloader) // 100, 1)
    outputs = []
    for idx, (inputs, prefixs, items) in enumerate(dataloader):
        assert len(inputs) == len(items)
        assert len(inputs) == len(prefixs)
        assert len(inputs) == 1

        cnt = args.num_return_sequences - item2num[(str(items[0]['image']), items[0]['question'])]
        if cnt <= 0:
            continue

        response_list = []
        for _ in range(0, cnt, args.batch_size):
            prompts = [inputs[0]] * args.batch_size
            prompts_processed = []
            for prompt in prompts:
                if args.load_image:
                    prompt = pipe._convert_prompts(prompt)
                else:
                    prompt = pipe._convert_prompts(prompt.replace(IMAGE_TOKEN, '').strip())
                prompt.append({'role': 'prefix', 'content': prefixs[0]})
                prompts_processed.append(prompt)

            curr_response_list = pipe(prompts_processed, **generation_config)
            response_list.extend([prefixs[0] + response.text for response in curr_response_list])
        assert len(response_list) == cnt

        if len(inputs[0]) < 1:
            continue

        if args.load_image:
            query_list = [inputs[0][0]] * cnt
        else:
            query_list = [inputs[0]] * cnt

        for item_idx, item in enumerate(items):
            n = cnt
            for r in response_list[item_idx * n: item_idx * n + n]:
                item = item.copy()
                item['rejected'] = r
                outputs.append(item)

        if idx % log_freq == 0:
            print(
                f'[{localtime()}] '
                f'[Rank {torch.distributed.get_rank()}] '
                f'[Progress {idx}/{len(dataloader)}] '
            )

        if idx % print_freq == 0 and torch.distributed.get_rank() == 0:
            log_str = [
                f'[Prompt]\n{query_list[-1]}',
                f'[Image]\n{outputs[-1]["image"]}',
                f'[Input]\n{outputs[-1]["question"]}',
                f'[Chosen]\n{outputs[-1]["chosen"]}',
                f'[Prefix]\n{prefixs[-1]}',
                f'[Output]\n{outputs[-1]["rejected"]}',
                f'[Answer]\n{outputs[-1]["answer"]}' if 'answer' in outputs[-1] else '',
                f'[End]',
            ]
            print('\n'.join(log_str))

        if torch.distributed.get_rank() == 0:
            print(f'[Item {idx} {localtime()}] Finish to log')

    print(f'[{localtime()}] [Rank {torch.distributed.get_rank()}] Finish')

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, outputs)

    merged_outputs = sum(merged_outputs, start=[])

    if torch.distributed.get_rank() == 0:
        with open(results_file, 'a') as file:
            for output in merged_outputs:
                file.write(json.dumps(output) + '\n')

        print(f'[{localtime()}] Results ({len(merged_outputs)=}) saved to {results_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--prompt-path', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='sampled_outputs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--min-new-tokens', type=int, default=5)
    parser.add_argument('--num-return-sequences', type=int, default=8)
    parser.add_argument('--start-ratio', type=float, default=0.5)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--load-image', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--sample-max-num', type=int, default=None)
    args = parser.parse_args()

    assert args.num_return_sequences % args.batch_size == 0
    assert args.temperature > 0

    init_distributed_mode()

    # model_name = '_'.join(args.checkpoint.split('/')[-2:])
    # args.out_dir = os.path.join(args.out_dir, model_name)

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

    Qwen7BChat.messages2prompt = messages2prompt
    InternVL2InternLM2.messages2prompt = messages2prompt

    vision_config = VisionConfig(max_batch_size=25)
    pipe = pipeline(
        args.checkpoint,
        vision_config=vision_config,
        backend_config=TurbomindEngineConfig(session_len=8192, tp=args.tp)
    )
    pipe.vl_encoder.model.config.max_dynamic_patch = args.max_num
    pipe.vl_encoder.model.config.dynamic_image_size = args.dynamic

    assert isinstance(pipe.chat_template, (Qwen7BChat, InternVL2InternLM2))

    # lmdeploy will update the current_device
    torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    print(
        f'Begin to sample data from model {args.checkpoint}, '
        f'dynamic: {pipe.vl_encoder.model.config.dynamic_image_size}, '
        f'max_num: {pipe.vl_encoder.model.config.max_dynamic_patch}, '
    )
    evaluate_chat_model()
