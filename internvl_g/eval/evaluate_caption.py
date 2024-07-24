import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import torch
import torchvision.transforms as T
from internvl.model.internvl_stage2 import InternVLConfig, InternVLModel
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
    'coco': {
        'root': 'data/coco/',
        'annotation': ['data/coco/annotations/coco_karpathy_test.json',
                       'data/coco/annotations/coco_karpathy_test_gt.json'],
    },
    'nocaps': {
        'root': 'data/nocaps/images',
        'annotation': 'data/nocaps/nocaps_val_4500_captions.json',
    },
}


class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, name, root, annotation, prompt, input_size=224):
        if name == 'coco':
            self.images = json.load(open(annotation))
        else:
            self.images = json.load(open(annotation))['images']
        self.name = name
        self.prompt = prompt
        self.root = root
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

    config = InternVLConfig.from_pretrained(args.checkpoint)
    model = InternVLModel.from_pretrained(args.checkpoint, config=config).eval()
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
                    use_cache=True
                )
                image_ids.extend(ids)
                caption = [tokenizer.decode(_.cpu(), skip_special_tokens=True).strip() for _ in pred]
                captions.extend(caption)
                print(caption)

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='coco,flickr30k,nocaps')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

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

    evaluate_qllama_model()
