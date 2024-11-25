import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial
from typing import Literal

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'MME_RealWorld': {
        'root': 'InternVL-Domain-Adaptation-DataMME-RealWorld/val/MME_RealWorld.json',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'img_root': 'InternVL-Domain-Adaptation-DataMME-RealWorld/images/MME-RealWorld/data',
        'type': 'dev',
        'language': 'en'
    }
}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indexes = [_['index'] for _ in batches]
    choices = [_['choice'] for _ in batches]
    categorys = [_['category'] for _ in batches]
    tasks = [_['task'] for _ in batches]
    return pixel_values, questions, answers, indexes, choices, categorys, tasks


class MMERealworldDataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, language, subtask: Literal[
        'Monitoring', 'OCR with Complex Context', 'Diagram and Table', 'Autonomous_Driving', 'Remote Sensing'],
                 img_root, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        with open(root, 'r') as f:
            self.data_meta = json.load(f)
        self.subtask = subtask
        self.data_meta = [item for item in self.data_meta if item['Subtask'] == self.subtask]
        self.img_root = img_root
        self.prompt = prompt
        self.language = language
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data_meta)

    def __getitem__(self, idx):
        index = self.data_meta[idx]['Question_id']
        assert self.data_meta[idx]['Question Type'] == 'Multiple Choice'
        image = os.path.join(self.img_root, self.data_meta[idx]['Image'])
        question = self.data_meta[idx]['Text']
        choices = self.data_meta[idx]['Answer choices']
        answer = self.data_meta[idx]['Ground truth']
        category = self.data_meta[idx]['Category']
        task = self.data_meta[idx]['Task']
        # catetory = self.df.iloc[idx]['category']
        # l2_catetory = self.df.iloc[idx]['l2-category']

        image = Image.open(image).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        if self.language == 'cn':
            question = question + 'The choices are listed below:\n' + '\n'.join(choices) + '\n' + self.prompt['cn']
        else:
            question = question + '选项如下所示:\n' + '\n'.join(choices) + '\n' + self.prompt['en']

        return {
            'question': question,
            'pixel_values': pixel_values,
            'answer': answer,
            'index': index,
            'choice': choices,
            'category': category,
            'task': task
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


def post_process(s, choices):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:',
        'Best option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ''
    return matches[0]


def evaluate(outputs):
    results = {'Reasoning': {},
               'Perception': {}}
    for data_item in outputs:
        cnt = data_item['answer'] == data_item['gt_answers']
        category = data_item['category']
        task = data_item['task']
        if category not in results[task]:
            results[task][category] = {'true': cnt, 'false': 1 - cnt}
        else:
            results[task][category]['true'] += cnt
            results[task][category]['false'] += 1 - cnt

    cnt_subtask, sum_subtask = 0, 0
    for task, tasks_values in results.items():
        cnt_task, sum_task = 0, 0
        for category, category_dict in tasks_values.items():
            cnt_task += category_dict['true']
            sum_task += category_dict['false'] + category_dict['true']
            acc = category_dict['true'] / (category_dict['false'] + category_dict['true'])
            print(f'-' * 4 + f'\t' + 'Acc ' + '{:.4f}'.format(acc) + f'\t{category.capitalize()}')

        cnt_subtask += cnt_task
        sum_subtask += sum_task
        if sum_task == 0:
            acc_task = 0
        else:
            acc_task = cnt_task / sum_task
        print(f'*' * 32 + f'Acc' + '{:.4f}'.format(acc_task) + f'\t{task}')

    if sum_subtask == 0:
        acc_subtasks = 0
    else:
        acc_subtasks = cnt_subtask / sum_subtask
    print(f'+' * 16 + f'\t Acc ' + '{:.4f}'.format(acc_subtasks))
    return acc_subtasks


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MMERealworldDataset(
            root=ds_collections[ds_name]['root'],
            prompt=prompt,
            language=ds_collections[ds_name]['language'],
            subtask=args.subtask,
            img_root=ds_collections[ds_name]['img_root'],
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
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
        for pixel_values, questions, answers, indexes, options, categorys, tasks in tqdm(dataloader):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            out = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config
            )
            outs = [out]
            preds = [post_process(out, options[0])]

            for question, pred, answer, index, out, category, task in zip(questions, preds, answers, indexes, outs,
                                                                          categorys, tasks):
                outputs.append({
                    'question': question,
                    'output': out,
                    'answer': pred,
                    'gt_answers': answer,
                    'index': index,
                    'category': category,
                    'task': task
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
            results_file = f'{ds_name}_{args.subtask}_{time_prefix}.json'
            output_path = os.path.join(args.out_dir, results_file)

            with open(output_path, 'w') as f:
                json.dump(merged_outputs, f, indent=4)
            evaluate(merged_outputs)

            print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='MME_RealWorld')
    parser.add_argument('--subtask', type=str, default='Autonomous_Driving')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    prompt = {
        'en': 'Select the best answer to the above multiple-choice question based on the image. \
            Respond with only the letter (A, B, C, D, or E) of the correct option. \nThe best answer is:',
        'cn': '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n 最佳答案为：',
    }
    evaluate_chat_model()
