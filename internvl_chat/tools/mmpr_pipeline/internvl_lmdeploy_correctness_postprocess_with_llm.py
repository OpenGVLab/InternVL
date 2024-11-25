import os
import json
import time
import socket
import datetime
import argparse
import subprocess

import torch

from lmdeploy import TurbomindEngineConfig, pipeline


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


def check(item):
    question = item['question']
    response = item['response']
    answer = item['answer']

    prompt = PROMPT.format(question=question, gt=answer, pred=response)
    output = pipe([prompt]).text[0].strip().strip('.').strip().lower()

    if output == 'yes':
        is_correct = True
    elif output == 'no':
        is_correct = False
    else:
        print(
            f'[Start]'
            f'#Invalid output:\n{output}\n\n'
            f'[SEP]'
            f'#Current prompt:\n{prompt}\n\n'
            f'[End]'
        )
        is_correct = False

    return is_correct


def main():
    for filename in os.listdir(args.data_dir):
        if not filename.endswith('.jsonl'):
            continue

        with open(os.path.join(args.data_dir, filename)) as file:
            lines = file.readlines()

        items = []
        for line in lines:
            item = json.loads(line)
            item['is_correct'] = check(item)

        with open(os.path.join(args.save_dir, filename)) as file:
            for item in items:
                file.write(json.dumps(item) + '\n')


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

    init_dist(args)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    pipe = pipeline(
        args.judger,
        backend_config=TurbomindEngineConfig(session_len=32784, tp=args.tp),
    )
    torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    main()
    print(f'Finish')
