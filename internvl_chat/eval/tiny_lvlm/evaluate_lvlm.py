import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from internvl.train.dataset import build_transform
from PIL import Image
from tqdm import tqdm
from transformers import LlamaTokenizer

ds_collections = {
    'updated_datasets': {
        'root': 'data/tiny_lvlm/updated_datasets/',
        'max_new_tokens': 30,
        'min_new_tokens': 1,
    }
}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    image_paths = [_['image_path'] for _ in batches]

    return pixel_values, questions, annotations, image_paths


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, input_size=224, pad2square=False):
        dirnames = [os.path.join(root, item) for item in os.listdir(root)]
        dirnames = [item for item in dirnames if os.path.exists(os.path.join(item, 'dataset.json'))]
        sorted(dirnames)

        self.roots = []
        self.items = []
        for item in dirnames:
            data_path = os.path.join(item, 'dataset.json')
            data = json.loads(open(data_path).read())
            for data_line in data:
                self.roots.append(item)
                self.items.append(data_line)
        self.prompt = prompt
        self.transform = build_transform(is_train=False, input_size=input_size, pad2square=pad2square)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        root = self.roots[idx]
        item = self.items[idx]
        image_path, question, annotation = item['image_path'], item['question'], item['gt_answers']
        image_path = os.path.join(root, image_path)
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.transform(image).unsqueeze(0)
        question = question + ' ' + self.prompt
        return {
            'question': question,
            'pixel_values': pixel_values,
            'annotation': annotation,
            'image_path': image_path,
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
    prompt = 'Answer the question using a single word or phrase.'
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = VQADataset(
            root=ds_collections[ds_name]['root'],
            prompt=prompt,
            input_size=image_size,
            pad2square=pad2square
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (pixel_values, questions, annotations, image_paths) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                length_penalty=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config,
            )
            answers = [pred]

            for question, answer, annotation, image_path in zip(questions, answers, annotations, image_paths):
                task_type = image_path.split('/')[-2]
                outputs.append({
                    'question': question,
                    'answer': answer,
                    'gt_answers': annotation,
                    'image_path': image_path,
                    'task_type': task_type
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))
            cmd = 'python eval/tiny_lvlm/calculate_score.py ' \
                  '--file-path ' + results_file
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='updated_datasets')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)

    if 'qllama' in args.checkpoint.lower():
        from internvl.model.internvl_chat_with_qllama import InternVLChatModel
        model = InternVLChatModel.from_pretrained(
            args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).cuda().eval()
        image_size = model.internvl.config.force_image_size or model.config.internvl_config.vision_config.image_size
        pad2square = model.config.pad2square
    else:
        from internvl.model.internvl_chat import InternVLChatModel
        model = InternVLChatModel.from_pretrained(
            args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).cuda().eval()
        image_size = model.config.force_image_size or model.config.vision_config.image_size
        pad2square = model.config.pad2square

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 30:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] pad2square: {pad2square}')
    print(f'[test] template: {model.config.template}')

    evaluate_chat_model()
