import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from data_utils import CAT_SHORT2LONG, process_single_sample
from datasets import concatenate_datasets, load_dataset
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'MMMU_validation': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'validation'
    },
    'MMMU_test': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'test'
    },
    'MMMU_dev': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'dev'
    },
}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    options = [_['option'] for _ in batches]
    return pixel_values, questions, answers, data_ids, options


class MMMUDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        # run for each subject
        sub_dataset_list = []
        for subject in tqdm(CAT_SHORT2LONG.values()):
            sub_dataset = load_dataset(root, subject, split=split, cache_dir=os.path.join(os.getcwd(), 'data/MMMU/'))
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        self.data = concatenate_datasets(sub_dataset_list)
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = process_single_sample(self.data[idx])
        data_id = data['id']
        question = data['question'].strip()
        pil_images = data['image']
        question_type = data['question_type']

        choices = eval(data['options'])
        answer = data['answer'] if 'answer' in data else None

        choice_list = []
        options = {}
        multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c.strip()))
            options[multiple_choices[i]] = c.strip()
        choice_txt = '\n'.join(choice_list)
        if self.dynamic_image_size:
            images = []
            for idx, pil_image in enumerate(pil_images):
                if pil_image is not None:
                    if idx == 0:
                        pil_image = pil_image.resize((pil_image.width * 2, pil_image.height * 2), Image.BILINEAR)
                        pil_image = dynamic_preprocess(pil_image, image_size=self.input_size,
                                                       use_thumbnail=self.use_thumbnail, max_num=self.max_num)
                    else:
                        pil_image = dynamic_preprocess(pil_image, image_size=self.input_size,
                                                       use_thumbnail=self.use_thumbnail, max_num=1)
                    images += pil_image
        else:
            images = [pil_images[0]]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        if len(choice_txt) > 0:
            question += '\n' + choice_txt
        question += '\n' + self.prompt[question_type]
        question = question.strip()

        return {
            'question': question,
            'pixel_values': pixel_values,
            'answer': answer,
            'option': options,
            'data_id': data_id
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


def post_process(pred, option):
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k

    return pred


def evaluate_chat_model():
    prompt = {
        'multiple-choice': "Answer with the option's letter from the given choices directly.",
        'open': 'Answer the question using a single word or phrase.'
    }
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MMMUDataset(
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
        for _, (pixel_values, questions, answers, data_ids, options) in tqdm(enumerate(dataloader)):
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
                verbose=True
            )
            if len(options[0]) == 0:
                preds = [pred]
            else:
                preds = [post_process(pred, options[0])]

            for question, pred, answer, data_id in zip(questions, preds, answers, data_ids):
                outputs.append({
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'data_id': data_id
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
            output_path = os.path.join(args.out_dir, results_file)
            outputs = {}
            for item in merged_outputs:
                outputs[item['data_id']] = item['answer']
            with open(output_path, 'w') as f:
                json.dump(outputs, f, indent=4)
            print('Results saved to {}'.format(output_path))
            if ds_collections[ds_name]['split'] == 'validation':
                print('Evaluating ...')
                cmd = f'python eval/mmmu/main_eval_only.py ' \
                      f'--output_path {output_path} ' \
                      f'--answer_path eval/mmmu/answer_dict_val.json'
                print(cmd)
                os.system(cmd)
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.jsonl'
            output_path = os.path.join(args.out_dir, results_file)
            writer = open(output_path, 'w')
            for item in merged_outputs:
                writer.write(json.dumps(item) + '\n')
            writer.close()
            print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='MMMU_validation')
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
