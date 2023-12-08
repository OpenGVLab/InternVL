import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
import torchvision.transforms as T
from husky.model.internvl_hf_stage2_v7 import (InternQformerConfig,
                                               InternQformerModel)
from husky.model.internvl_hf_stage3_v13 import InternChatModel
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import LlamaTokenizer

ds_collections = {
    'flickr30k': {
        'root': 'data/flickr30k/',
        'annotation': 'data/flickr30k/flickr30k_test_karpathy.json',
    },
    'flickr30k-qwen': {
        'root': '',
        'annotation': 'data/flickr30k/flickr30k_karpathy_test.json',
    },
    'coco': {
        'root': 'data/coco/',
        'annotation': ['data/coco/annotations/coco_karpathy_test.json',
                       'data/coco/annotations/coco_karpathy_test_gt.json'],
    },
    'nocaps': {
        'root': 'data/nocaps/images',
        'annotation': 'data/nocaps/nocaps_val_4500_captions.json',
    },
    'nocaps-qwen': {
        'root': '',
        'annotation': 'data/nocaps/nocaps_val.json',
    },
}


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, name, root, annotation, prompt, input_size=224, pad2square=False):
        if name == 'coco':
            self.images = json.load(open(annotation))
        else:
            self.images = json.load(open(annotation))['images']
        self.name = name
        self.prompt = prompt
        self.root = root
        if pad2square:
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.name == 'coco':
            filename = self.images[idx]['image']
            image_id = int(filename.split('_')[-1].replace('.jpg', ''))
            image_path = os.path.join(self.root, filename)
        else:
            image_id = self.images[idx]['id']
            if 'file_name' in self.images[idx]:
                image_path = os.path.join(self.root, self.images[idx]['file_name'])
            else:
                image_path = os.path.join(self.root, self.images[idx]['image'])
        image = Image.open(image_path)
        pixel_values = self.transform(image).unsqueeze(0)
        return {
            'image_id': image_id,
            'input_text': self.prompt,
            'pixel_values': pixel_values
        }


def collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts, return_tensors='pt')

    return pixel_values, image_ids, input_tokens.input_ids, input_tokens.attention_mask


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


def evaluate_qllama_model():
    prompts = ['English caption:']
    print('prompts:', prompts)

    config = InternQformerConfig.from_pretrained(args.checkpoint)
    model = InternQformerModel.from_pretrained(args.checkpoint, config=config).eval()
    model = model.to(torch.float16).cuda()
    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
    tokenizer.add_eos_token = False

    random.seed(args.seed)
    summaries = []
    for prompt in prompts:
        for ds_name in args.datasets:
            annotation = ds_collections[ds_name]['annotation']
            if type(annotation) == list:
                annotation = annotation[0]
            if model.config.force_image_size is not None:
                image_size = model.config.force_image_size
            else:
                image_size = model.config.vision_config.image_size
            dataset = CaptionDataset(
                name=ds_name,
                root=ds_collections[ds_name]['root'],
                annotation=annotation,
                prompt=prompt,
                input_size=image_size,
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

            image_ids, captions = [], []
            for _, (pixel_values, ids, input_ids, attention_mask) in tqdm(enumerate(dataloader)):
                pred = model.generate(
                    pixel_values=pixel_values.cuda().to(torch.float16),
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    do_sample=False,
                    num_beams=args.num_beams,
                    max_new_tokens=30,
                    min_new_tokens=8,
                    use_cache=False
                )
                image_ids.extend(ids)
                captions.extend([
                    tokenizer.decode(_.cpu(), skip_special_tokens=True).strip() for _ in pred
                ])

            torch.distributed.barrier()

            world_size = torch.distributed.get_world_size()
            merged_ids = [None for _ in range(world_size)]
            merged_captions = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(merged_ids, image_ids)
            torch.distributed.all_gather_object(merged_captions, captions)

            merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
            merged_captions = [_ for _ in itertools.chain.from_iterable(merged_captions)]
            average_length = sum(len(x.split()) for x in merged_captions) / len(merged_captions)
            print(f'Average length: {average_length}')

            if torch.distributed.get_rank() == 0:
                print(f'Evaluating {ds_name} ...')

                results = []
                for image_id, caption in zip(merged_ids, merged_captions):
                    results.append({
                        'image_id': int(image_id),
                        'caption': caption,
                    })
                time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
                results_file = f'{ds_name}_{time_prefix}.json'
                results_file = os.path.join(args.out_dir, results_file)
                json.dump(results, open(results_file, 'w'))

                annotation = ds_collections[ds_name]['annotation']
                if type(annotation) == list:
                    annotation = annotation[-1]
                coco = COCO(annotation)
                coco_result = coco.loadRes(results_file)
                coco_eval = COCOEvalCap(coco, coco_result)
                coco_eval.evaluate()

                summary = coco_eval.eval.items()
                print([ds_name, prompt, average_length, summary])
                summaries.append([ds_name, prompt, average_length, summary])

            torch.distributed.barrier()

    for summary in summaries:
        print(summary)


def evaluate_chat_model():
    question = 'Provide a one-sentence caption for the provided image.'
    print('question:', question)

    model = InternChatModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.checkpoint)
    tokenizer.add_eos_token = False

    random.seed(args.seed)
    summaries = []
    for ds_name in args.datasets:
        annotation = ds_collections[ds_name]['annotation']
        if type(annotation) == list:
            annotation = annotation[0]
        if model.qllama.config.force_image_size is not None:
            image_size = model.qllama.config.force_image_size
        else:
            image_size = model.qllama.config.vision_config.image_size
        dataset = CaptionDataset(
            name=ds_name,
            root=ds_collections[ds_name]['root'],
            annotation=annotation,
            prompt=question,
            input_size=image_size,
            pad2square=args.pad2square
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

        image_ids, captions = [], []
        for _, (pixel_values, ids, _, _) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                do_sample=False,
                num_beams=args.num_beams,
                max_new_tokens=30,
                min_new_tokens=8,
            )
            pred = model.chat(
                template=args.template,
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
            )
            image_ids.extend(ids)
            captions.extend([pred])

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_ids = [None for _ in range(world_size)]
        merged_captions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_ids, image_ids)
        torch.distributed.all_gather_object(merged_captions, captions)

        merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
        merged_captions = [_ for _ in itertools.chain.from_iterable(merged_captions)]
        average_length = sum(len(x.split()) for x in merged_captions) / len(merged_captions)
        print(f'Average length: {average_length}')

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')

            results = []
            for image_id, caption in zip(merged_ids, merged_captions):
                results.append({
                    'image_id': int(image_id),
                    'caption': caption,
                })
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(results, open(results_file, 'w'))

            annotation = ds_collections[ds_name]['annotation']
            if type(annotation) == list:
                annotation = annotation[-1]
            coco = COCO(annotation)
            coco_result = coco.loadRes(results_file)
            coco_eval = COCOEvalCap(coco, coco_result)
            coco_eval.evaluate()

            summary = coco_eval.eval.items()
            print(summary)
            summaries.append([args.checkpoint, ds_name, average_length, summary])

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
    parser.add_argument('--datasets', type=str, default='coco,flickr30k,nocaps')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--template', type=str, default='vicuna_v1.1')
    parser.add_argument('--model_type', type=str, choices=['InternQformerModel', 'InternChatModel'])
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pad2square', type=bool, default=False)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    print('pad2square:', args.pad2square)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if args.model_type == 'InternQformerModel':
        evaluate_qllama_model()
    elif args.model_type == 'InternChatModel':
        evaluate_chat_model()
    else:
        raise NotImplementedError
