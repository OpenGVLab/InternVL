import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'DriveLM_val': {
        'root': 'InternVL-Domain-Adaptation-Data/val/drivelm_val.jsonl',
        'max_new_tokens': 200,
        'min_new_tokens': 1,
        'split': 'validation',
        'image_root': 'InternVL-Domain-Adaptation-Data/images/drivelm/stitch',
    }
}


def post_process(pred):
    pred = pred.strip()
    pattern = r'<c[^,]*,\s*[^,]*,\s*\[\s*-?[0-9]*\.?[0-9]+\s*,\s*-?[0-9]*\.?[0-9]+\s*\]\s*>'
    mapping = {'CAM_FRONT_LEFT': [0, 0], 'CAM_FRONT': [1, 0], 'CAM_FRONT_RIGHT': [2, 0], 'CAM_BACK_LEFT': [0, 1],
               'CAM_BACK': [1, 1], 'CAM_BACK_RIGHT': [2, 1]}
    patch_size = 448
    width = patch_size * 2
    height = patch_size
    whole_img_width = width * 3
    whole_img_height = height * 2
    matches = re.findall(pattern, pred)
    for object_id in matches:

        object_id_c = object_id.replace('<', '').replace('>', '')
        try:
            ctag = object_id_c.split(',')[0]
            cxcy = json.loads(','.join(object_id_c.split(',')[2:]))
            cam = object_id_c.split(',')[1]
            if cam in mapping:
                mx, my = mapping[cam]
                # old_wide,old_height = images_size[cam]
                old_wide, old_height = 1600, 900
                cx, cy = cxcy
                cx = (cx / 1000) * whole_img_width
                cy = (cy / 1000) * whole_img_height
                cx -= mx * width
                cy -= my * height
                cx = cx / width * old_wide
                cy = cy / height * old_height
                # cx =max(0,min(old_wide,cx))
                # cy =max(0,min(old_height,cy))
                cx = round(max(0, min(old_wide, cx)), 1)
                cy = round(max(0, min(old_height, cy)), 1)
                new_object_id = f'<{ctag},{cam},{cx},{cy}>'

                pred = pred.replace(object_id, new_object_id)
        except Exception as e:
            print(e)
    return pred


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    questions_old = [_['question_old'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    return pixel_values, questions_old, questions, answers, data_ids


class DriveLMDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, prompt, image_path, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, ):
        with open(root, 'r') as f:
            self.data = [json.loads(line) for line in f.readlines()]
            # data_val = json.load(f)
        # merge all dataset
        # self.data = concatenate_datasets(sub_dataset_list)
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.image_path = image_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]
        data_id = data['id']
        question = data['conversations'][0]['value'].strip()
        question_old = data['question_old']
        image_file = os.path.join(self.image_path, data['image'])
        image = Image.open(image_file).convert('RGB')
        # question_type = data['question_type']
        # choices = eval(data['options'])
        answer = data['conversations'][1]['value'].strip()

        if self.dynamic_image_size:
            pil_image = dynamic_preprocess(image, image_size=self.input_size,
                                           use_thumbnail=self.use_thumbnail,
                                           max_num=self.max_num)
            images = pil_image
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'question_old': question_old,
            'question': question,
            'pixel_values': pixel_values,
            'answer': answer,
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


def evaluate_chat_model():
    random.seed(args.seed)
    prompt = None
    for ds_name in args.datasets:
        dataset = DriveLMDataset(
            root=ds_collections[ds_name]['root'],
            split=ds_collections[ds_name]['split'],
            prompt=prompt,
            image_path=ds_collections[ds_name]['image_root'],
            # image_meta = ds_collections[ds_name]["image_meta"],
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
        for _, (pixel_values, questions_old, questions, answers, data_ids) in tqdm(enumerate(dataloader)):
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

            preds = [post_process(pred)]

            for question, pred, answer, data_id, question_old in zip(questions, preds, answers, data_ids,
                                                                     questions_old):
                outputs.append({
                    'question': question_old,
                    'answer': pred,
                    'gt_answers': answer,
                    'id': data_id
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
            print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='DriveLM_val')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=12)
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
