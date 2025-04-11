import os

if os.environ.get('AUTO_SPLIT', '0') == '1':
    TP = int(os.environ.get('TP', '4'))
    DEVICE_START_IDX = int(os.environ['SLURM_PROCID']) % 8
    CUDA_VISIBLE_DEVICES = [str(i) for i in range(DEVICE_START_IDX, DEVICE_START_IDX + TP)]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(CUDA_VISIBLE_DEVICES)
    print(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")
else:
    TP = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ['SLURM_PROCID']) % 8)
    print(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")

import argparse
import io
import json
from collections import defaultdict

import torch
from lmdeploy import (ChatTemplateConfig, GenerationConfig,
                      TurbomindEngineConfig, VisionConfig, pipeline)
from lmdeploy.vl.constants import IMAGE_TOKEN
from PIL import Image
from tools.reasoning_data_pipeline.utils.constants import IMG_PLACEHOLDER
from tools.reasoning_data_pipeline.utils.utils import (InferenceSampler,
                                                       get_global_min,
                                                       init_dist, localtime,
                                                       save_outputs)

try:
    from petrel_client.client import Client
    client = Client()
except:
    import socket
    import warnings
    ip = socket.gethostbyname(socket.gethostname())
    warnings.warn(
        f'[{ip}] Fail to import petrel_client! '
        f'You can ignore this warning if you do not need to load image from ceph.'
    )


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def collate_fn(batches):
    items = []
    inputs = []
    prefixs = []
    for batch in batches:
        items.append(batch['item'])
        if 'image' in batch:
            inputs.append((batch['question'], batch['image']))
        else:
            inputs.append(batch['question'])
        prefixs.append(batch['prefix'])

    return inputs, prefixs, items


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        start_ratio=0.5,
        sample_max_num=None,
        load_image=False,
    ):
        with open(data) as file:
            self.data = file.readlines()

        self.start_ratio = start_ratio
        self.load_image = load_image

        if sample_max_num is not None and len(self.data) > sample_max_num:
            print(f'Truncate data lines. {len(self.data)} => {sample_max_num}')
            step = len(self.data) // sample_max_num
            self.data = self.data[args.sample_start_idx::step][:sample_max_num]

    def __len__(self):
        return len(self.data)

    def _truncate_prefix(self, prefix):
        splitted_prefix = prefix.split(' ')
        sep_idx = int(len(splitted_prefix) * self.start_ratio)
        splitted_prefix = splitted_prefix[:sep_idx]
        return ' '.join(splitted_prefix).strip()

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        question = item['question']
        prefix = self._truncate_prefix(item['chosen'])

        if self.load_image:
            images = item['image']
            if not isinstance(images, (list, tuple)):
                images = [images]

            images_new = []
            for image in images:
                if 's3://' in image:
                    image = io.BytesIO(client.get(image))
                image = Image.open(image).convert('RGB')
                images_new.append(image)
            images = images_new

            return {
                'question': question.replace(IMG_PLACEHOLDER, IMAGE_TOKEN),
                'image': images,
                'prefix': prefix,
                'item': item.copy(),
            }

        return {
            'question': question.replace(IMG_PLACEHOLDER, IMAGE_TOKEN),
            'prefix': prefix,
            'item': item.copy(),
        }


def evaluate_chat_model():
    dataset = VQADataset(
        data=args.prompt_path,
        start_ratio=args.start_ratio,
        sample_max_num=args.sample_max_num,
        load_image=args.load_image,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        sampler=InferenceSampler(len(dataset)),
    )
    min_len = get_global_min(len(dataloader))

    item2num = defaultdict(int)
    results_file = os.path.basename(args.prompt_path)
    results_file = os.path.join(args.out_dir, results_file)
    if os.path.exists(results_file):
        with open(results_file) as file:
            lines = file.readlines()
        for line in lines:
            item = json.loads(line)
            if args.load_image:
                item2num[(str(item['image']), item['question'], item['chosen'])] += 1
            else:
                item2num[(item['question'], item['chosen'])] += 1

    print(
        f'[Rank {torch.distributed.get_rank()}] '
        f'Begin to answer {len(dataloader)} batches '
        f'(about {len(dataloader) * args.batch_size} samples), '
        f'{args.prompt_path=}, '
        f'{len(item2num)=}'
    )

    log_freq = max(len(dataloader) // args.batch_size // 100, 1)
    print_freq = max(len(dataloader) // args.batch_size // 100, 1)
    outputs = []
    for idx, (inputs, prefixs, items) in enumerate(dataloader):
        assert len(inputs) == len(items)
        assert len(inputs) == len(prefixs)

        cnt_list = []
        filtered_inputs = []
        filtered_items = []
        filtered_prefixs = []
        for i in range(len(inputs)):
            if args.load_image:
                key = (str(items[i]['image']), items[i]['question'], items[i]['chosen'])
            else:
                key = (items[i]['question'], items[i]['chosen'])
            cnt = args.num_return_sequences - item2num[key]
            if cnt <= 0:
                continue
            cnt_list.append(cnt)
            filtered_inputs.append(inputs[i])
            filtered_items.append(items[i])
            filtered_prefixs.append(prefixs[i])

        inputs = filtered_inputs
        items = filtered_items
        prefixs = filtered_prefixs

        if len(inputs) <= 0:
            print(f'skip {idx}')
            if idx % log_freq == 0 and idx < min_len:
                save_outputs(outputs, results_file)
                outputs = []
            continue

        for _ in range(max(cnt_list)):
            gen_config.random_seed = None
            prompts_processed = []
            for prompt_idx, prompt in enumerate(inputs):
                if args.load_image:
                    prompt = pipe._convert_prompts(prompt)
                else:
                    prompt = pipe._convert_prompts(prompt.replace(IMAGE_TOKEN, '').strip())
                prompt.append({'role': 'assistant', 'content': prefixs[prompt_idx]})
                prompts_processed.append(prompt)

            response_list = pipe(prompts_processed, gen_config=gen_config)
            response_list = [prefixs[0] + response.text for response in response_list]

            for prefix, item, response in zip(prefixs, items, response_list):
                response = prefix + response.text

                item = item.copy()
                item['rejected'] = response
                outputs.append(item)

        if idx % print_freq == 0 and torch.distributed.get_rank() == 0:
            log_str = [
                f'[Prompt]\n{inputs[-1][0] if isinstance(inputs[-1], tuple) else inputs[-1]}\n'
                f'[Image]\n{outputs[-1]["image"]}',
                f'[Input]\n{outputs[-1]["question"]}',
                f'[Chosen]\n{outputs[-1]["chosen"]}',
                f'[Prefix]\n{prefixs[-1]}',
                f'[Output]\n{outputs[-1]["rejected"]}',
                f'[Answer]\n{outputs[-1]["answer"]}' if 'answer' in outputs[-1] else '',
                f'[End]',
            ]
            print('\n'.join(log_str))

        if idx % log_freq == 0:
            print(
                f'[{localtime()}] '
                f'[Rank {torch.distributed.get_rank()}] '
                f'[Progress {idx}/{len(dataloader)}] '
            )

        if idx % log_freq == 0 and idx < min_len:
            save_outputs(outputs, results_file)
            outputs = []

    print(f'[{localtime()}] [Rank {torch.distributed.get_rank()}] Finish')

    save_outputs(outputs, results_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base args
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--prompt-path', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='sampled_outputs')
    parser.add_argument('--num-workers', type=int, default=8)
    # lmdeploy args
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--vit-batch-size', type=int, default=8)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--session-len', type=int, default=16384)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.3)
    # generation args
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    # sampling args
    parser.add_argument('--load-image', action='store_true')
    parser.add_argument('--start-ratio', type=float, default=0.5)
    parser.add_argument('--num-return-sequences', type=int, default=1)
    parser.add_argument('--sample-start-idx', type=int, default=0)
    parser.add_argument('--sample-max-num', type=int, default=None)
    args = parser.parse_args()
    args.tp = TP

    assert args.temperature > 0

    init_dist(args)

    model_name = '_'.join(args.checkpoint.split('/')[-2:])
    args.out_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(args.out_dir, exist_ok=True)

    gen_config = GenerationConfig(
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    vision_config = VisionConfig(max_batch_size=args.vit_batch_size)
    pipe = pipeline(
        args.checkpoint,
        vision_config=vision_config,
        chat_template_config=ChatTemplateConfig(model_name='internvl2_5'),
        backend_config=TurbomindEngineConfig(session_len=args.session_len, cache_max_entry_count=args.cache_max_entry_count, tp=args.tp),
    )
    pipe.vl_encoder.model.config.max_dynamic_patch = args.max_num
    pipe.vl_encoder.model.config.dynamic_image_size = args.dynamic

    # lmdeploy will update the current_device
    torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    print(
        f'Begin to sample data from model {args.checkpoint}, '
        f'dynamic: {pipe.vl_encoder.model.config.dynamic_image_size}, '
        f'max_num: {pipe.vl_encoder.model.config.max_dynamic_patch}, '
        f'sample_start_idx: {args.sample_start_idx}, '
    )
    evaluate_chat_model()
