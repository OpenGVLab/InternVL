import io
import os
import json
import time
import socket
import random
import datetime
import argparse
import subprocess

import torch

from PIL import Image
from collections import defaultdict
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from petrel_client.client import Client


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


client = Client()
INSTRUCTION = (
    "### Question:\n{question}\n\n"
    "### Answer:\n{answer}\n\n"
    "### Analysis:\n{analysis}\n\n"
    "Your task is to write a step-by-step solution to the given question, referring to the provided analysis and answer. "
    "After you finish the solution, please write \"Final answer: xxx\" in the last line.\n\n"
    "If the given question is a multi-choice question, the \"xxx\" should be an option's letter. (e.g., Final answer: A)\n"
    "If the given question is a mathematics question, the \"xxx\" refers to the final numeric answer. (e.g. Final answer: 0)"
)
INSTRUCTION = INSTRUCTION.strip()


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        sample_max_num=None,
    ):
        with open(data) as file:
            self.data = file.readlines()

        if sample_max_num is not None and len(self.data) > sample_max_num:
            print(f'Truncate data lines. {len(self.data)} => {sample_max_num}')
            step = len(self.data) // sample_max_num
            self.data = self.data[::step]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        question = item['question']
        answer_gt = item['answer']
        analysis = item['analysis']

        question = INSTRUCTION.format(question=question, analysis=analysis, answer=answer_gt)

        return {
            'question': question,
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


def collate_fn(batches):
    items = []
    inputs = []
    for batch in batches:
        items.append(batch['item'])
        inputs.append(batch['question'])

    return inputs, items


def localtime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def merge_lines(lines):
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, lines)

    merged_outputs = sum(merged_outputs, start=[])
    return merged_outputs


def get_global_min(value):
    world_size = torch.distributed.get_world_size()
    merged_values = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_values, value)

    return min(merged_values)


def parse_answer(response):
    answer_trigger = 'Final answer:'
    if response.count(answer_trigger) == 0:
        answer_trigger = 'Final Answer:'

    assert response.count(answer_trigger) <= 2, f"Fail to find Answer, {response.count(answer_trigger)=}"
    assert response.count('\n') >= 2, f"Fail to find rationale, {response=}"

    rationale, answer = response.rsplit(answer_trigger, 1)
    assert len(rationale.strip()) > 0, f"Empty rationale:\n{response}"
    assert '\n' not in answer.strip(), f"Answer with multiple paragraphs:\n{answer}"

    return rationale.strip(), answer.strip()


def annotate_lines(dataloader, existed, save_path):
    err_cnt = 0
    items_with_ann = []

    min_len = get_global_min(len(dataloader))
    log_freq = max(len(dataloader) // args.bsz // 100, 1)

    for idx, (inputs, items) in enumerate(dataloader):
        new_inputs = []
        new_items = []
        for input, item in zip(inputs, items):
            question = item['question']
            answer_gt = item['answer']
            analysis = item['analysis']

            if (question, answer_gt, analysis) in existed:
                continue

            new_inputs.append(input)
            new_items.append(item)

        inputs = new_inputs
        items = new_items

        if len(inputs) == 0:
            continue

        outputs = pipe(
            inputs,
            gen_config=gen_config,
        )
        outputs = [output.text for output in outputs]

        for output, item in zip(outputs, items):
            try:
                _, answer_parsed = parse_answer(output)
                assert answer_parsed in answer_gt, f'Incorrect parsed answer: {answer_parsed=}, {answer_gt=}'
            except Exception as e:
                err_cnt += 1
                print(e)
                print()
                print(output)
                print()
                continue

            item_with_ann = item.copy()
            item_with_ann['response'] = output
            item_with_ann['answer'] = answer_parsed
            item_with_ann['answer_orig'] = answer_gt
            items_with_ann.append(item_with_ann)

        if idx % log_freq == 0:
            print(
                f'[Progress] [{localtime()}] [Rank {rank}] [{idx}/{len(dataloader)}] [{err_cnt}/{(idx+1)*args.bsz}]'
            )

        if idx % log_freq == 0 and rank == 0 and len(items_with_ann) > 0:
            print(
                f'[Question]\n'
                f'{items_with_ann[-1]["question"]}\n'
                f'[Analysis]\n'
                f'{items_with_ann[-1]["analysis"]}\n'
                f'[Response]\n'
                f'{items_with_ann[-1]["response"]}\n'
                f'[Answer]\n'
                f'{items_with_ann[-1]["answer"]}\n'
                f'[Answer_orig]\n'
                f'{items_with_ann[-1]["answer_orig"]}\n'
            )

        if idx % log_freq == 0 and idx < min_len:
            save_lines(items_with_ann, save_path)
            items_with_ann = []

    print(f'[Rank {rank}] Finish')

    if rank == 0:
        print(
            f'[Progress] [{localtime()}] {len(dataloader)} {len(items_with_ann)=}'
        )
    return items_with_ann


def save_lines(items, save_path):
    items = merge_lines(items)

    if rank != 0:
        return

    with open(save_path, 'a') as file:
        for item in items:
            file.write(json.dumps(item) + '\n')
    print(f'Save {len(items)} lines in {save_path}')


def init_distributed_mode():
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = rank % torch.cuda.device_count()

    world_size = int(os.environ["SLURM_NTASKS"])
    local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

    if "MASTER_PORT" not in os.environ:
        port = 22222
        # for i in range(22222, 65535):
        #     cmd = f'netstat -aon|grep {i}'
        #     with os.popen(cmd, 'r') as file:
        #         if '' == file.read():
        #             port = i
        #             break

        print(f'MASTER_PORT = {port}')
        os.environ["MASTER_PORT"] = str(port)

        time.sleep(3)

    node_list = os.environ["SLURM_NODELIST"]
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr

    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(local_size)
    os.environ['WORLD_SIZE'] = str(world_size)

    torch.cuda.set_device(local_rank)


def init_dist(args):
    init_distributed_mode()

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


def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    dataset = VQADataset(
        data=args.data_path,
        sample_max_num=args.max_lines,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bsz,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        sampler=InferenceSampler(len(dataset)),
    )

    existed = set()
    filename = os.path.basename(args.data_path)
    lines_save_path = os.path.join(save_dir, filename)
    if os.path.exists(lines_save_path):
        with open(lines_save_path) as file:
            lines = file.readlines()

        for line in lines:
            item = json.loads(line)
            question = item['question']
            answer_gt = item['answer']
            analysis = item['analysis']

            existed.add((question, answer_gt, analysis))

    print(f'preprocess {filename}, {len(dataloader)=}, {args.bsz=}')
    items_with_ann = annotate_lines(
        dataloader=dataloader,
        existed=existed,
        save_path=lines_save_path,
    )

    save_lines(items_with_ann, lines_save_path)
    print()


if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--data-path', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='')
    parser.add_argument('--max-lines', type=int, default=int(1e6))
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--bsz', type=int, default=1)
    args = parser.parse_args()
    args.tp = 8

    init_dist(args)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    pipe = pipeline(
        args.checkpoint,
        backend_config=TurbomindEngineConfig(session_len=16384, tp=args.tp),
    )
    gen_config = GenerationConfig(
        temperature=0.7,
        max_new_tokens=4096,
    )

    # lmdeploy will update the current_device
    torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    main(args)
    print('Finish')
