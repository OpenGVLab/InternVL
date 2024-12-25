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
    'mpdocvqa_val': {
        'root': 'data/mpdocvqa/images/',
        'test': 'data/mpdocvqa/val.json',
        'annotation': 'data/mpdocvqa/val.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'mpdocvqa_test': {
        'root': 'data/mpdocvqa/images/',
        'test': 'data/mpdocvqa/test.json',
        'metric': None,
        'max_new_tokens': 100,
    }
}


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    num_patches_lists = [_['num_patches_list'] for _ in batches]

    return pixel_values, questions, question_ids, annotations, num_patches_lists


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, root, test, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, total_max_num=64):
        self.root = root
        self.test = json.loads(open(test).read())['data']
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.total_max_num = total_max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test[idx]
        page_ids = data['page_ids']
        question_id = data['questionId']
        question = data['question']
        annotation = data.get('answers', None)
        image_list = []
        for page_id in page_ids:
            image_path = os.path.join(self.root, page_id + '.jpg')
            image = Image.open(image_path).convert('RGB')
            image_list.append(image)

        max_num = max(1, min(self.max_num, self.total_max_num // len(image_list)))
        num_patches_list = []
        if self.dynamic_image_size:
            images = []
            for image in image_list:
                tiles = dynamic_preprocess(image, image_size=self.input_size,
                                           use_thumbnail=self.use_thumbnail,
                                           max_num=max_num)
                images += tiles
                num_patches_list.append(len(tiles))
        else:
            images = image_list
            num_patches_list.append(1)
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        if len(images) > 1:
            prefix = ''.join([f'Image-{i + 1}: <image>\n' for i in range(len(image_list))])
            question = prefix + question
        if len(self.prompt) != 0:
            question = question + ' ' + self.prompt
        return {
            'question_id': question_id,
            'question': question,
            'pixel_values': pixel_values,
            'annotation': annotation,
            'num_patches_list': num_patches_list
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
    base_prompt = 'Answer the question using a single word or phrase.'
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        dataset = VQADataset(
            root=ds_collections[ds_name]['root'],
            test=ds_collections[ds_name]['test'],
            prompt=base_prompt,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            total_max_num=args.total_max_num,
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
        for _, (pixel_values, questions, question_ids, annotations, num_patches_lists) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            with torch.inference_mode():
                pred = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config,
                    num_patches_list=num_patches_lists[0],
                    verbose=True
                )
                torch.cuda.empty_cache()
            answers = [pred]

            for question, question_id, answer, annotation in zip(questions, question_ids, answers, annotations):
                if ds_name in ['mpdocvqa_val']:
                    outputs.append({
                        'question': question,
                        'questionId': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['mpdocvqa_test']:
                    outputs.append({
                        'questionId': question_id,
                        'answer': answer,
                    })
                else:
                    raise NotImplementedError

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

            if ds_collections[ds_name]['metric'] == 'anls':
                json.dump(merged_outputs,
                          open(results_file, 'w'),
                          ensure_ascii=False)
                print('python eval/mpdocvqa/infographicsvqa_eval.py -g ' +
                      ds_collections[ds_name]['annotation'] + ' -s ' +
                      results_file)
                os.system('python eval/mpdocvqa/infographicsvqa_eval.py -g ' +
                          ds_collections[ds_name]['annotation'] + ' -s ' +
                          results_file)

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='mpdocvqa_val')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=18)
    parser.add_argument('--total-max-num', type=int, default=64)
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
