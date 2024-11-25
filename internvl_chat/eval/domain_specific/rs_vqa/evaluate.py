import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'RSVQA_H_TEST2': {
        'root': 'InternVL-Domain-Adaptation-Data/val/rsvqa_h_test_2_instruct.json',
        'max_new_tokens': 50,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': 'InternVL-Domain-Adaptation-Data/images/RSVQA-H/Data'
    },
    'RSVQA_H_TEST1': {
        'root': 'InternVL-Domain-Adaptation-Data/val//rsvqa_h_test_1_instruct.json',
        'max_new_tokens': 50,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': 'InternVL-Domain-Adaptation-Data/images/RSVQA-H/Data'
    },
    'RSVQA_L': {
        'root': 'InternVL-Domain-Adaptation-Data/val/rsvqa_l_test_instruct.json',
        'max_new_tokens': 50,
        'min_new_tokens': 1,
        'type': 'test',
        'image_root': 'InternVL-Domain-Adaptation-Data/images/RSVQA_L/Images_LR'
    },
}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indexes = [_['index'] for _ in batches]
    question_types = [_['question_type'] for _ in batches]

    return pixel_values, questions, answers, indexes, question_types


class RSVQADataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, image_root, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):

        with open(root, 'r') as f:
            self.ann_data = json.load(f)
        self.prompt = prompt
        self.image_root = image_root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.ann_data)

    def __getitem__(self, idx):
        data_item = self.ann_data[idx]
        index = data_item['id']
        image = data_item['image']
        question = data_item['question'] + '\n' + self.prompt
        answer = data_item['gt_answer']
        question_type = data_item['type']
        # catetory = self.df.iloc[idx]['category']
        # l2_catetory = self.df.iloc[idx]['l2-category']
        image = Image.open(os.path.join(self.image_root, image)).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'question': question,
            'pixel_values': pixel_values,
            'answer': answer,
            'index': index,
            'question_type': question_type
        }


def evaluation_metrics(outputs):
    correct = 0
    incorrect = 0
    for output in outputs:
        gt = output['gt_answers']
        answer = output['answer'].split(',')[0].lower().replace('.', '')
        if gt == answer:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
        # else:
        #     continue
    print('correct:', correct)
    print('incorrect:', incorrect)
    print('Total:', correct + incorrect)
    print('Acc:', (correct / (correct + incorrect)))

    return {
        'correct:': correct,
        'incorrect:': incorrect,
        'Total:': correct + incorrect,
        'Acc:': correct / (correct + incorrect)
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
        dataset = RSVQADataset(
            root=ds_collections[ds_name]['root'],
            prompt=prompt,
            image_root=ds_collections[ds_name]['image_root'],
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
        for _, (pixel_values, questions, answers, indexes, question_types) in tqdm(enumerate(dataloader)):
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
                generation_config=generation_config
            )
            preds = [pred]

            for question, pred, answer, index, question_type in zip(questions, preds, answers, indexes, question_types):
                outputs.append({
                    'question': question,
                    'response': pred,
                    'gt_answer': answer,
                    'index': int(index),
                    'question_type': question_type
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
            with open(output_path, 'w') as f:
                json.dump(merged_outputs, f, indent=4)
            cmd = f'python eval/rs_vqa/score.py --output_file {output_path}'
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='RSVQA_H_TEST2')
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

    prompt = 'Answer the question using a single word or phrase.'
    evaluate_chat_model()
