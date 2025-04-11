import datetime
import json
import os
import socket
import subprocess
import time

import torch


def localtime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def init_distributed_mode():
    if 'RANK' in os.environ:
        print('Found existed environs, skip slurm envs init for dist')
        return

    rank = int(os.environ['SLURM_PROCID'])
    local_rank = rank % torch.cuda.device_count()

    world_size = int(os.environ['SLURM_NTASKS'])
    # local_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
    local_size = torch.cuda.device_count()

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

    try:
        torch.cuda.set_device(local_rank)
    except:
        print(f'Error ip: {socket.gethostbyname(socket.gethostname())}')
        raise


def init_dist(args):
    init_distributed_mode()

    if getattr(args, 'port', None) is not None:
        os.environ['MASTER_PORT'] = str(args.port)

    if int(os.getenv('RANK', '0')) % args.tp != 0:
        print(f"[SLURM_PROCID {int(os.environ['SLURM_PROCID'])}] Exit early")
        exit(0)

    if args.tp > 1:
        os.environ['RANK'] = str(int(os.environ['RANK']) // args.tp)
        os.environ['LOCAL_RANK'] = str(int(os.environ['RANK']) % args.tp)
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

    try:
        torch.distributed.barrier()
    except:
        print(f'Error ip: {socket.gethostbyname(socket.gethostname())}')
        raise

    print(
        f'world_size={torch.distributed.get_world_size()}, '
        f'rank={torch.distributed.get_rank()}, '
        f'ip={socket.gethostbyname(socket.gethostname())}, '
        f'device_count={torch.cuda.device_count()}, '
    )


def get_global_min(value):
    world_size = torch.distributed.get_world_size()
    merged_values = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_values, value)

    return min(merged_values)


def save_outputs(outputs, results_file):
    if 'image' in outputs[0]:
        outputs = sorted(outputs, key=lambda x:str(x['image']))
    else:
        outputs = sorted(outputs, key=lambda x:str(x['question']))

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, outputs)

    merged_outputs = sum(merged_outputs, start=[])

    if torch.distributed.get_rank() == 0:
        with open(results_file, 'a') as file:
            for output in merged_outputs:
                file.write(json.dumps(output) + '\n')

        print(f'[{localtime()}] Results ({len(merged_outputs)=}) saved to {results_file}')


def load_outputs(results_file):
    with open(results_file) as file:
        lines = file.readlines()
    items = [json.loads(line) for line in lines]
    return items


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
