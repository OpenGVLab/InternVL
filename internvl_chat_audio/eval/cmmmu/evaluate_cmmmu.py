import argparse
import json
import os
import random

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'art_and_design': {
        'root': 'data/',
        'annotation': 'data/cmmmu-data/llava_art_and_design.jsonl',
        'max_new_tokens': 999,
        'min_new_tokens': 1,
    },
    'business': {
        'root': 'data/',
        'annotation': 'data/cmmmu-data/llava_business.jsonl',
        'max_new_tokens': 999,
        'min_new_tokens': 1,
    },
    'health_and_medicine': {
        'root': 'data/',
        'annotation': 'data/cmmmu-data/llava_health_and_medicine.jsonl',
        'max_new_tokens': 999,
        'min_new_tokens': 1,
    },
    'humanities_and_social_sciences': {
        'root': 'data/',
        'annotation': 'data/cmmmu-data/llava_humanities_and_social_sciences.jsonl',
        'max_new_tokens': 999,
        'min_new_tokens': 1,
    },
    'science': {
        'root': 'data/',
        'annotation': 'data/cmmmu-data/llava_science.jsonl',
        'max_new_tokens': 999,
        'min_new_tokens': 1,
    },
    'technology_and_engineering': {
        'root': 'data/',
        'annotation': 'data/cmmmu-data/llava_technology_and_engineering.jsonl',
        'max_new_tokens': 999,
        'min_new_tokens': 1,
    },
}


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, root, annotation, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.root = root
        self.items = []
        f = open(annotation)
        data = f.readlines()
        for data_line in data:
            data_line = json.loads(data_line)
            self.items.append(data_line)
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path, question = item['image'], item['text']
        image_path = os.path.join(self.root, image_path)
        image = Image.open(image_path).convert('RGB')
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
            'item': item,
        }


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = VQADataset(
            root=ds_collections[ds_name]['root'],
            annotation=ds_collections[ds_name]['annotation'],
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )

        print(f'Evaluating {ds_name} ...')
        results_file = f'{model_id}_{ds_name}.jsonl'
        results_file = os.path.join(args.out_dir, results_file)
        writer = open(results_file, 'w')

        for _, data in tqdm(enumerate(dataset)):
            pixel_value = data['pixel_values']
            question = data['question']
            item = data['item']
            pixel_value = pixel_value.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_value,
                question=question,
                generation_config=generation_config,
                verbose=True
            )
            question_id = item['question_id']
            text = item['text']
            output = {
                'question_id': question_id,
                'prompt': text,
                'text': pred,
                'model_id': model_id,
                'metadata': {}
            }
            writer.write(json.dumps(output, ensure_ascii=False) + '\n')
            writer.flush()
        print('Results saved to {}'.format(results_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')
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
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

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

    model_id = '_'.join(args.checkpoint.split('/')[-2:])
    evaluate_chat_model()
