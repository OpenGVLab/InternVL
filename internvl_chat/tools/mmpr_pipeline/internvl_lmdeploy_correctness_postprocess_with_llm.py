import os
import json
import time
import socket
import datetime
import argparse
import subprocess

import torch

from collections import defaultdict
from lmdeploy import TurbomindEngineConfig, GenerationConfig, pipeline


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


PROMPT = """You are an expert in verifying the reasoning process based on a question-answer pair. We asked an examiner to answer a question about a picture.

[Start of Question]

{question}

[End of Question]

[Start of GT Answer]

{gt}

[End of GT Answer]

[Start of Examiner's Answer]

{pred}

[End of Examiner's Answer]


Please identify whether the given reasoning process and answer are correct and consistent with the ground truth answer. You should only answer yes or no.
""".strip()


def localtime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def collate_fn(batches):
    items = []
    inputs = []
    for batch in batches:
        items.append(batch['item'])
        inputs.append(PROMPT.format(question=batch['question'], gt=batch['answer'], pred=batch['response']))

    return inputs, items


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
        response = item['response']
        answer = item['answer']

        return {
            'question': question,
            'response': response,
            'answer': answer,
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


def check(items, outputs):
    for item, output in zip(items, outputs):
        if output == 'yes':
            is_correct = True
        elif output == 'no':
            is_correct = False
        else:
            print(
                f'[Start]'
                f'#Invalid output:\n{output}\n\n'
                f'[End]'
            )
            is_correct = False

        item['is_correct'] = is_correct

    return items


def save_outputs(outputs, results_file):
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


def main():
    for i in [6, 12, 18, 24]:
        data_dir = os.path.join(args.data_dir, f'max_tiles_{i}')
        if not os.path.exists(data_dir):
            continue

        save_dir = os.path.join(args.save_dir, f'max_tiles_{i}')
        os.makedirs(save_dir, exist_ok=True)

        for filename in os.listdir(data_dir):
            if not filename.endswith('.jsonl'):
                continue

            dataset = VQADataset(
                data=os.path.join(data_dir, filename),
                sample_max_num=args.max_lines,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False,
                num_workers=8,
                sampler=InferenceSampler(len(dataset)),
            )

            gen_config = GenerationConfig(
                temperature=0,
                max_new_tokens=10,
            )

            item2exist = defaultdict(bool)
            save_path = os.path.join(save_dir, filename)
            if os.path.exists(save_path):
                with open(save_path) as file:
                    lines = file.readlines()

                for line in lines:
                    item = json.loads(line)
                    item2exist[(item['question'], item['answer'], item['response'])] = item.get('is_correct', None) is not None

            log_freq = max(len(dataloader) // args.batch_size // 100, 1)

            items_list = []
            statistics = defaultdict(int)
            for idx, (inputs, items) in enumerate(dataloader):
                filtered_items = []
                filtered_inputs = []
                for i in range(len(inputs)):
                    exist = item2exist[(items[i]['question'], items[i]['answer'], items[i]['response'])]
                    if exist:
                        continue
                    filtered_items.append(items[i])
                    filtered_inputs.append(inputs[i])

                items = filtered_items
                inputs = filtered_inputs

                if len(inputs) <= 0:
                    continue

                outputs = pipe(inputs, gen_config=gen_config)
                outputs = [
                    output.text.strip().strip('.').strip().lower()
                    for output in outputs
                ]
                items = check(items, outputs)
                items_list.extend(items)

                for item in items:
                    statistics['total'] += 1
                    statistics['positive'] += item['is_correct']

                if idx % log_freq == 0:
                    print(
                        f'[{localtime()}] '
                        f'[Rank {torch.distributed.get_rank()}] '
                        f'[Progress {idx}/{len(dataloader)}] '
                        f"{statistics['total']=}, {statistics['positive']=}"
                    )
                    save_outputs(items_list, save_path)
                    items_list = []

            save_outputs(items_list, save_path)


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


def init_dist():
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--judger', type=str, default='')
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='')
    parser.add_argument('--max-lines', type=int, default=int(1e6))
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    init_dist()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    pipe = pipeline(
        args.judger,
        backend_config=TurbomindEngineConfig(session_len=32784, tp=args.tp),
    )
    torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    main()
    print(f'Finish')
