import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import numpy as np
import torch
from datasets import load_dataset
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from tqdm import tqdm

ds_collections = {
    'Mantis-Eval': {
        'root': 'TIGER-Lab/Mantis-Eval',
        'max_new_tokens': 50,
        'min_new_tokens': 1,
        'split': 'test'
    },
}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    num_patches_lists = [_['num_patches_list'] for _ in batches]
    options = [_['option'] for _ in batches]
    lines = [_['line'] for _ in batches]
    return pixel_values, questions, answers, num_patches_lists, options, lines


class MantisEvalDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        dataset = load_dataset(root, split=split, cache_dir=os.path.join(os.getcwd(), 'data/mantis_eval/'))
        self.data = dataset
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        question_type = data['question_type']
        question = data['question']
        image_list = data['images']
        del data['images']
        del data['data_source']
        images_to_remove = ' '.join(['<image>'] * len(image_list))
        question = question.replace(images_to_remove, '').strip()
        for i in range(len(image_list)):
            question = question.replace('<image>', f'Image-{i + 1}', 1)
        options = data['options']
        options = [item.strip() for item in options]
        answer = data['answer']
        choice_txt = '\n'.join(options)

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
        pixel_values = torch.stack(pixel_values)

        if question_type == 'multi-choice':
            question = question + '\n' + choice_txt + '\n' + self.prompt[question_type]
        else:
            question = question + '\n' + self.prompt[question_type]
        question = question.strip()
        if len(image_list) == 1:
            prefix = '<image>\n'
        else:
            prefix = ''.join([f'Image-{i + 1}: <image>\n' for i in range(len(image_list))])
        question = prefix + question
        return {
            'question': question,
            'pixel_values': pixel_values,
            'answer': answer,
            'option': options,
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
    prompt = {
        'multi-choice': "Answer with the option's letter from the given choices directly.",
        'short-answer': 'Answer the question using a single word or phrase.'
    }
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MantisEvalDataset(
            root=ds_collections[ds_name]['root'],
            split=ds_collections[ds_name]['split'],
            prompt=prompt,
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
        for _, (pixel_values, questions, answers, num_patches_lists, options, lines) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config,
                num_patches_list=num_patches_lists[0],
                verbose=True
            )
            preds = [pred]
            for question, pred, answer, line in zip(questions, preds, answers, lines):
                line['question'] = question
                line['pred'] = pred
                line['answer'] = answer
                options = line['options']
                question_type = line['question_type']
                if question_type == 'multi-choice':
                    if len(pred) == 3 and pred[0] == '(' and pred[-1] == ')':
                        pred = pred[1:-1]
                    if pred == options[ord(answer) - ord('A')] or pred == answer:
                        line['correct'] = 1
                    else:
                        line['correct'] = 0
                else:
                    if pred.lower() == answer.lower():
                        line['correct'] = 1
                    else:
                        line['correct'] = 0
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
            for item in merged_outputs:
                writer.write(json.dumps(item) + '\n')
            writer.close()
            print('Results saved to {}'.format(output_path))

            multi_choice_items = [item for item in merged_outputs if item['question_type'] == 'multi-choice']
            if len(multi_choice_items) > 0:
                print(f'Multi-choice Accuracy: {np.mean([q["correct"] for q in multi_choice_items]):.4f}')

            open_ended_items = [item for item in merged_outputs if item['question_type'] == 'short-answer']
            if len(open_ended_items) > 0:
                print(f'Short-answer Accuracy: {np.mean([q["correct"] for q in open_ended_items]):.4f}')

            if len(merged_outputs) > 0:
                print(f"Overall Accuracy: {np.mean([q['correct'] for q in merged_outputs]):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='Mantis-Eval')
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
