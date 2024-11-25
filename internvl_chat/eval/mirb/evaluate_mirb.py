import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial

import numpy as np
import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'MIRB': {
        'root': 'data/MIRB',
        'max_new_tokens': 512,
        'min_new_tokens': 1,
        'split': 'test'
    },
}

word_to_num = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
    'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10'
}

# Define the groupings
groups = {
    'Knowledge': ['food', 'sightseeing'],
    'Reasoning': ['codeu', 'plot_code', 'analogy', '3d_scene'],
    'Perception': ['image_jigsaw', 'count', 'attribute'],
    'Multi-Hop': ['visual_chain', 'arxiv']
}


def eval_scores(results, dataset):
    if dataset in ['count', 'codeu', 'food', 'image_jigsaw', 'arxiv', 'visual_chain', 'visual_chain_concat',
                   'plot_code', '3d_scene', '3d_scene_concat', 'count_concat', 'image_needles',
                   'image_needles_concat', 'plot_text', 'arxiv_text', 'codeu_text']:
        score = exact_match(results, dataset)
    elif dataset in ['analogy', 'attribute']:
        score = exact_yes_no(results)
    elif dataset in ['sightseeing']:
        score = exact_in_match(results)
    return score


def exact_yes_no(results):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if result['answers'].lower() == 'yes' and 'yes' in str(prediction).lower():
            acc.append(1)
        elif result['answers'].lower() == 'no' and 'yes' not in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc


def exact_in_match(results):
    acc = []
    for result in results:
        if result['answers'].lower() in ['yes', 'no']:
            prediction = result['prediction'].strip()
            prediction = prediction.strip('\n')
            trunc_index = prediction.find('\n')
            if trunc_index <= 0:
                trunc_index = prediction.find('.')
            if trunc_index > 0:
                prediction = prediction[:trunc_index]
            if result['answers'].lower() == 'yes' and 'yes' in str(prediction).lower():
                acc.append(1)
            elif result['answers'].lower() == 'no' and 'yes' not in str(prediction).lower():
                acc.append(1)
            else:
                acc.append(0)
            continue
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if str(result['answers']).lower() in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc


def exact_match(results, dataset):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if dataset in ['count', 'count_concat',
                       'visual_chain', 'visual_chain_concat',
                       '3d_scene', '3d_scene_concat',
                       'image_needles', 'image_needles_concat']:
            # find the number
            match = re.search(r'\d+', prediction)
            if match:
                prediction = match.group()
            else:
                if str(prediction.lower()) in word_to_num:
                    prediction = word_to_num[str(prediction.lower())]
                else:
                    prediction = ''

        if str(prediction).lower() == str(result['answers']).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc


def collate_fn(batches, tokenizer):
    try:
        pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    except:
        pixel_values = [None for _ in batches]
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    num_patches_lists = [_['num_patches_list'] for _ in batches]
    lines = [_['line'] for _ in batches]
    return pixel_values, questions, answers, num_patches_lists, lines


def get_task_instruction(dataset):
    if dataset in ['analogy', 'attribute', 'plot_code', 'plot_text', 'sightseeing',
                   'image_needles', 'image_needles_concat', 'visual_chain', 'visual_chain_concat']:
        instr = 'Answer with a single word.'
    elif dataset in ['codeu', 'food', 'image_jigsaw', 'codeu_text']:
        instr = 'Answer with the option symbol.'
    elif dataset in ['arxiv', 'arxiv_text']:
        instr = 'Answer with the paper title.'
    elif dataset in ['count', 'count_concat']:
        instr = 'Answer with a single number.'
    elif dataset in ['3d_scene', '3d_scene_concat']:
        instr = 'The following images are different views of the same 3D scene. Answer with a single number.'
    return instr


class MIRBDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        json_files = [item for item in os.listdir(root) if item.endswith('.json')]
        self.data = []
        for json_file in json_files:
            json_path = os.path.join(root, json_file)
            # if '_concat' in json_path:
            #     continue
            task = os.path.basename(json_path).replace('.json', '')
            temp = json.loads(open(json_path).read())
            for item in temp:
                item['task'] = task
                self.data.append(item)
        self.root = root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        questions = data['questions']
        answers = str(data['answers'])
        task = data['task']
        prompt = get_task_instruction(task)
        question = questions + '\n' + prompt
        input_image_path = data['images']
        image_list = []
        for image_path in input_image_path:
            image_path = os.path.join(self.root, image_path)
            image = Image.open(image_path).convert('RGB')
            image_list.append(image)

        num_patches_list = []
        if self.dynamic_image_size:
            images = []
            for image in image_list:
                tiles = dynamic_preprocess(image, image_size=self.input_size,
                                           use_thumbnail=self.use_thumbnail,
                                           max_num=max(1, self.max_num // len(image_list)))
                images += tiles
                num_patches_list.append(len(tiles))
        else:
            images = image_list
            num_patches_list.append(1)
        pixel_values = [self.transform(image) for image in images]
        if len(pixel_values) > 0:
            pixel_values = torch.stack(pixel_values)
        else:
            pixel_values = None

        if len(image_list) == 1:
            prefix = '<image>\n'
        else:
            prefix = ''.join([f'Image-{i + 1}: <image>\n' for i in range(len(image_list))])
        question = prefix + question
        return {
            'question': question,
            'pixel_values': pixel_values,
            'answer': answers,
            'num_patches_list': num_patches_list,
            'line': data
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
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MIRBDataset(
            root=ds_collections[ds_name]['root'],
            split=ds_collections[ds_name]['split'],
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
        for _, (pixel_values, questions, answers, num_patches_lists, lines) in tqdm(enumerate(dataloader)):
            try:
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
            except:
                pixel_values = None
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            try:
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config,
                    num_patches_list=num_patches_lists[0],
                    verbose=False
                )
            except:
                pred = 'Error'
            preds = [pred]
            for question, pred, answer, line in zip(questions, preds, answers, lines):
                task = line['task']
                line = {
                    'question': question,
                    'prediction': pred,
                    'answers': answer,
                    'task': task
                }
                score = eval_scores([line], task)
                line['score'] = score
                outputs.append(line)

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.jsonl'
            output_path = os.path.join(args.out_dir, results_file)
            writer = open(output_path, 'w')
            scores = {}
            for item in merged_outputs:
                task = item['task']
                score = item['score']
                if task not in scores:
                    scores[task] = []
                scores[task].append(score)
                writer.write(json.dumps(item) + '\n')
            writer.close()
            print('Results saved to {}'.format(output_path))
            averages = {}
            for group_name, group_list in groups.items():
                values = [np.mean(scores[task]) for task in group_list]
                averages[group_name] = np.mean(values)
            for category in averages:
                print(f'{category} Acc: {averages[category]}')
            print(f'Mean Acc: {np.mean(list(averages.values()))}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='MIRB')
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

    evaluate_chat_model()
