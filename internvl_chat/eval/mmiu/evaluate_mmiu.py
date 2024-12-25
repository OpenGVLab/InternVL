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
    'mmiu': {
        'root': 'data/mmiu',
        'annotation': 'eval/mmiu/mmiu.jsonl',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
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


class MMIUDataset(torch.utils.data.Dataset):

    def __init__(self, meta, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        # run for each subject
        meta_path = meta['annotation']
        f = open(meta_path, 'r')
        lines = f.readlines()
        self.data = [json.loads(line) for line in lines]
        self.root = meta['root']
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        input_image_path = data['input']['input_image_path']
        if len(input_image_path) == 1:
            image_prefix = ''
        else:
            image_cnt = len(input_image_path)
            image_prefix = ''.join([f'Image-{i+1}: <image>\n' for i in range(image_cnt)])
        question = data['input']['question']
        context = data['input']['context']
        question = image_prefix + question + '\n' + context + "\nAnswer with the option's letter from the given choices directly."

        input_image_path = [os.path.join(self.root, item) for item in input_image_path]
        image_list = []
        for image_path in input_image_path:
            image = Image.open(image_path).convert('RGB')
            image_list.append(image)

        options = data['options'].split('\n')
        answer = data['output']['output_text']

        new_options = {}
        multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                            'H', 'I', 'J', 'K', 'L', 'M', 'N',
                            'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                            'V', 'W', 'X', 'Y', 'Z']
        for i, c in enumerate(options):
            c = c.strip()
            if c.startswith(f'{multiple_choices[i]}:'):
                c = c.replace(f'{multiple_choices[i]}:', '')
            new_options[multiple_choices[i]] = c.strip()

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

        return {
            'question': question,
            'pixel_values': pixel_values,
            'answer': answer,
            'option': new_options,
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
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MMIUDataset(
            meta=ds_collections[ds_name],
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
            with torch.inference_mode():
                try:
                    pred = model.chat(
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=questions[0],
                        generation_config=generation_config,
                        num_patches_list=num_patches_lists[0],
                        verbose=True
                    )
                except:
                    print('Out of memory, skip this batch')
                    pred = 'A'
                torch.cuda.empty_cache()
            preds = [post_process(pred, options[0])]

            for question, pred, answer, line in zip(questions, preds, answers, lines):
                outputs.append({
                    'image': line['input']['input_image_path'],
                    'question': question,
                    'pred': pred,
                    'gt': answer,
                    'task': line['task'],
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
            results_file = f'{ds_name}_{time_prefix}.jsonl'
            output_path = os.path.join(args.out_dir, results_file)
            writer = open(output_path, 'w')

            acc_dict = {}
            for item in merged_outputs:
                writer.write(json.dumps(item) + '\n')
                task = item['task']
                pred = item['pred']
                gt = item['gt']

                if task not in acc_dict:
                    acc_dict[task] = []
                if pred == gt:
                    acc_dict[task].append(1)
                else:
                    acc_dict[task].append(0)
            writer.close()
            print('Results saved to {}'.format(output_path))
            orders = ['point_tracking', 'ravens_progressive_matrices', 'single_object_tracking',
                      'threed_cad_recognition', 'threed_indoor_recognition', 'Egocentric_Video_QuestionAnswering',
                      'Homography_estimation', 'Icon_Question_Answering_with_Spatial_Context',
                      'Image_Captioning_with_Spatial_Context', 'Image_Spatial_Transformation_Estimation',
                      'Image_text_retrieval_with_Spatial_Context', 'Multiview_Action_Recognition',
                      'Multiview_reasoning', 'jigsaw_puzzle_solving', 'threeD_Depth_Estimation',
                      'threeD_Object_Detection', 'threeD_Object_Tracking', 'threeD_Pose_Estimation',
                      'threeD_Scene_Reconstruction', 'threeD_question_answering']
            scores = []
            for task in orders:
                acc = acc_dict[task]
                num_correct = sum(acc)
                num_total = len(acc)
                print(f'{task} accuracy: {num_correct/num_total}')
                scores.append(num_correct/num_total)
            print(f'Overall accuracy: {sum(scores)/len(scores)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='mmiu')
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
