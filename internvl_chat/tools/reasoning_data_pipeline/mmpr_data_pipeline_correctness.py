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
from tools.reasoning_data_pipeline.utils.constants import (
    IMG_PLACEHOLDER, INSTRUCTION_BOXED_EN, INSTRUCTION_BOXED_ZH,
    INSTRUCTION_EN, INSTRUCTION_R1_EN, INSTRUCTION_R1_ZH, INSTRUCTION_ZH,
    VALID_INSTRUCTIONS)
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

    has_image = 'image' in batches[0]

    for batch in batches:
        assert ('image' in batch) == has_image

        items.append(batch['item'])
        if has_image:
            inputs.append((batch['question'], batch['image']))
        else:
            inputs.append(batch['question'])

    return inputs, items


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        sample_max_num=None,
    ):
        with open(data) as file:
            lines = file.readlines()

        self.data = []
        for line in lines:
            item = json.loads(line)
            self.data.append(line)

        if sample_max_num is not None and len(self.data) > sample_max_num:
            print(f'Truncate data lines. {len(self.data)} => {sample_max_num}')
            step = len(self.data) // sample_max_num
            self.data = self.data[args.sample_start_idx::step][:sample_max_num]

    def __len__(self):
        return len(self.data)

    def multi_modal_get_item(self, item):
        question = item['question']
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

        for instruction in VALID_INSTRUCTIONS:
            if question.endswith(instruction):
                question = question[:-len(instruction)].strip()
        # question = INSTRUCTION.format(question=question)
        question = INSTRUCTION.replace('{question}', question)

        if question.count(IMG_PLACEHOLDER) == 1:
            question = question.replace(IMG_PLACEHOLDER + '\n', '')
            question = question.replace(IMG_PLACEHOLDER, '')

        if question.count(IMG_PLACEHOLDER) == 0:
            question = IMG_PLACEHOLDER + '\n' + question

        return {
            'question': question.replace(IMG_PLACEHOLDER, IMAGE_TOKEN),
            'image': images,
            'item': item.copy(),
        }

    def pure_text_get_item(self, item):
        question = item['question']

        for instruction in VALID_INSTRUCTIONS:
            if question.endswith(instruction):
                question = question[:-len(instruction)].strip()
        # question = INSTRUCTION.format(question=question)
        question = INSTRUCTION.replace('{question}', question)

        return {
            'question': question.replace(IMG_PLACEHOLDER, IMAGE_TOKEN),
            'item': item.copy(),
        }

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        if 'image' in item and item['image']:
            return self.multi_modal_get_item(item)
        return self.pure_text_get_item(item)


def evaluate_chat_model():
    dataset = VQADataset(
        data=args.prompt_path,
        sample_max_num=args.sample_max_num,
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
            if 'image' in item and item['image']:
                key = (str(item['image']), item['question_orig'])
            else:
                key = item['question_orig']
            item2num[key] += 1

    print(
        f'[{localtime()}] [Rank {torch.distributed.get_rank()}] '
        f'Begin to answer {len(dataloader)} batches '
        f'(about {len(dataloader) * args.batch_size * args.num_return_sequences} samples), '
        f'{args.prompt_path=}, '
        f'{len(item2num)=}'
    )

    log_freq = max(len(dataloader) // args.batch_size // 100, 1)
    print_freq = max(len(dataloader) // args.batch_size // 100, 1)
    outputs = []
    for idx, (inputs, items) in enumerate(dataloader):
        assert len(inputs) == len(items)

        cnt_list = []
        filtered_items = []
        filtered_inputs = []
        for i in range(len(inputs)):
            if 'image' in items[i] and items[i]['image']:
                key = (str(items[i]['image']), items[i]['question'])
            else:
                key = items[i]['question']
            cnt = args.num_return_sequences - item2num[key]
            if cnt <= 0:
                continue
            cnt_list.append(cnt)
            filtered_items.append(items[i])
            filtered_inputs.append(inputs[i])

        items = filtered_items
        inputs = filtered_inputs

        if len(inputs) <= 0:
            print(f'skip {idx}')
            if idx % log_freq == 0 and idx < min_len:
                save_outputs(outputs, results_file)
                outputs = []
            continue

        for _ in range(max(cnt_list)):
            gen_config.random_seed = None
            response_list = pipe(inputs, gen_config=gen_config)

            for input, item, response in zip(inputs, items, response_list):
                item = item.copy()
                item['question_orig'] = item['question']
                if isinstance(input, str):
                    item['question'] = input
                else:
                    item['question'] = input[0].replace(IMAGE_TOKEN, IMG_PLACEHOLDER)
                item['response'] = response.text
                item['prompt_version'] = args.prompt_version
                outputs.append(item)

        if idx % print_freq == 0 and torch.distributed.get_rank() == 0:
            print(
                f'[Prompt]\n{inputs[-1][0] if isinstance(inputs[-1], tuple) else inputs[-1]}\n'
                # f'[Image]\n{outputs[-1]["image"]}\n'
                f'[Question]\n{outputs[-1]["question_orig"]}\n'
                f'[Output]\n{outputs[-1]["response"]}\n'
                f'[Answer]\n{outputs[-1]["answer"]}\n'
                f'[End]\n'
            )

        if idx % log_freq == 0:
            print(
                f'[{localtime()}] '
                f'[Rank {torch.distributed.get_rank()}] '
                f'[Progress {idx}/{len(dataloader)}] '
            )

        if idx % log_freq == 0 and idx < min_len:
            save_outputs(outputs, results_file)
            outputs = []

    print(f'[{localtime()}] [Rank {torch.distributed.get_rank()}] Finish to generate')

    save_outputs(outputs, results_file)

    print(f'[{localtime()}] [Rank {torch.distributed.get_rank()}] Finish to save outputs')


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
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    # sampling args
    parser.add_argument('--num-return-sequences', type=int, default=32)
    parser.add_argument('--sample-start-idx', type=int, default=0)
    parser.add_argument('--sample-max-num', type=int, default=None)
    parser.add_argument('--prompt-version', type=str, default=None, choices=['en', 'en_v2', 'zh', 'zh_v2', 'en_r1', 'zh_r1'])
    args = parser.parse_args()
    args.tp = TP

    if args.prompt_version is None:
        raise RuntimeError('Please set prompt_version')

    if '_zh_' in args.prompt_path:
        print(f'Set prompt_version to zh for {args.prompt_path}')
        args.prompt_version = args.prompt_version.replace('en', 'zh')

    global INSTRUCTION
    if args.prompt_version == 'zh':
        INSTRUCTION = INSTRUCTION_ZH
    elif args.prompt_version == 'zh_v2':
        INSTRUCTION = INSTRUCTION_BOXED_ZH
    elif args.prompt_version == 'zh_r1':
        INSTRUCTION = INSTRUCTION_R1_ZH
    elif args.prompt_version == 'en':
        INSTRUCTION = INSTRUCTION_EN
    elif args.prompt_version == 'en_v2':
        INSTRUCTION = INSTRUCTION_BOXED_EN
    elif args.prompt_version == 'en_r1':
        INSTRUCTION = INSTRUCTION_R1_EN
    else:
        assert False, f'Unsupported prompt version {args.prompt_version}'

    assert args.temperature > 0

    init_dist(args)

    model_name = '_'.join(args.checkpoint.split('/')[-2:])
    args.out_dir = os.path.join(args.out_dir, model_name, f'max_tiles_{args.max_num}')
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
        f'session_len: {args.session_len}, '
        f'dynamic: {pipe.vl_encoder.model.config.dynamic_image_size}, '
        f'max_num: {pipe.vl_encoder.model.config.max_dynamic_patch}, '
        f'sample_start_idx: {args.sample_start_idx}, '
    )
    evaluate_chat_model()
