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
import math
from collections import defaultdict

import torch
from lmdeploy import (ChatTemplateConfig, GenerationConfig,
                      TurbomindEngineConfig, VisionConfig, pipeline)
from lmdeploy.vl.constants import IMAGE_TOKEN
from PIL import Image
from tools.reasoning_data_pipeline.utils.accuracy_reward import (check_answer,
                                                                 get_mode,
                                                                 parse_answer)
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
    for batch in batches:
        items.append(batch['item'])
        inputs.append((batch['question'], batch['image']))

    return inputs, items


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        sample_max_num=None,
    ):
        with open(data) as file:
            self.data = file.readlines()

        if sample_max_num is not None and len(self.data) > sample_max_num:
            print(f'Truncate data lines. {len(self.data)} => {sample_max_num}')
            step = max(len(self.data) // sample_max_num, 1)
            self.data = self.data[args.sample_start_idx::step][:sample_max_num]
            print(f'Number of data lines after truncation: {len(self.data)=}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
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


def split_response(response, sep='\n\n', max_steps=None):
    steps = response.split(sep)

    if max_steps is not None:
        step = math.ceil(len(steps) / max_steps)
        new_steps = []
        for i in range(0, len(steps), step):
            new_steps.append(sep.join(steps[i:i+step]))
        return new_steps

    return steps


def join_steps(steps, sep='\n\n'):
    return sep.join(steps)


def build_responses(inputs, num_return_sequences=1, prefixes=None):
    messages_list = []
    for input_idx, input in enumerate(inputs):
        prompt, images = input

        content = [{'type': 'text', 'text': prompt}]
        for image in images:
            content.append({
                'type': 'image_data',
                'image_data': {
                    'data': image
                },
            })

        messages = [{
            'role': 'user',
            'content': content,
        }]

        if prefixes is not None:
            messages.append({
                'role': 'assistant',
                'content': prefixes[input_idx],
            })

        messages_list.append(messages)

    batched_response_list = [[] for _ in range(len(inputs))]
    for _ in range(num_return_sequences):
        gen_config.random_seed = None
        response_list = pipe(messages_list, gen_config=gen_config)
        response_list = [response.text for response in response_list]

        for response_idx, response in enumerate(response_list):
            if prefixes is not None:
                response = f'{prefixes[response_idx]}{response}'
            batched_response_list[response_idx].append(response)

    return sum(batched_response_list, start=[])


def build_mc_scores(inputs, response_list, items, num_return_sequences):
    assert len(response_list) == len(inputs) * num_return_sequences

    steps_list = [split_response(response, max_steps=args.max_steps) for response in response_list]
    steps_flag = [False for _ in range(len(response_list))]
    steps_outputs = [[] for _ in range(len(response_list))]

    step_cnt = 0
    while True:
        curr_inputs_idx = []
        curr_inputs = []
        curr_prefixes = []
        curr_answer_gt = []
        for idx, (steps, flag) in enumerate(zip(steps_list, steps_flag)):
            if step_cnt >= len(steps):
                continue

            if flag:
                steps_outputs[idx].append({
                    'step': steps[step_cnt],
                    'score': 0.0,
                    'num_mc_correct': 0,
                    'num_mc_total': 0,
                })
                continue

            input = inputs[idx // num_return_sequences]
            item = items[idx // num_return_sequences]

            curr_inputs_idx.append(idx)
            curr_inputs.append(input)
            curr_prefixes.append(join_steps(steps[:step_cnt+1]))
            curr_answer_gt.append(item['answer'])

        if len(curr_inputs) <= 0:
            for idx, steps in enumerate(steps_list):
                for step_idx in range(len(steps) - step_cnt - 1):
                    steps_outputs[idx].append({
                        'step': steps[step_cnt + step_idx + 1],
                        'score': 0.0,
                        'num_mc_correct': 0,
                        'num_mc_total': 0,
                    })
            break

        mc_response_list = build_responses(curr_inputs, args.num_mc_sequences, curr_prefixes)
        correctness_list = []
        for mc_idx, mc_response in enumerate(mc_response_list):
            try:
                correctness = check_answer(
                    answer_pred=parse_answer(mc_response, prompt_version=args.prompt_version)[-1],
                    answer_gt=curr_answer_gt[mc_idx // args.num_mc_sequences],
                    mode=args.verification_mode
                )
            except:
                print('Fail to check correctness for response:', mc_response)
                correctness = 0
            correctness_list.append(correctness)

        assert len(mc_response_list) == len(correctness_list)
        assert len(mc_response_list) == len(curr_inputs) * args.num_mc_sequences

        for idx_idx, idx in enumerate(curr_inputs_idx):
            curr_correctness_list = correctness_list[idx_idx*args.num_mc_sequences:(idx_idx+1)*args.num_mc_sequences]
            score = sum(curr_correctness_list) / len(curr_correctness_list)
            steps_outputs[idx].append({
                'step': steps_list[idx][step_cnt],
                'score': score,
                'num_mc_correct': sum(curr_correctness_list),
                'num_mc_total': len(curr_correctness_list),
            })

            if score == 0 and args.early_stop:
                steps_flag[idx] = True

        step_cnt += 1

    return steps_outputs


def build_process_supervision(inputs, items, num_return_sequences):
    response_list = build_responses(inputs, num_return_sequences)
    steps_with_score = build_mc_scores(inputs, response_list, items, num_return_sequences)

    outputs = []

    for idx, (response, each_steps_with_score) in enumerate(zip(response_list, steps_with_score)):
        input = inputs[idx // num_return_sequences]
        item = items[idx // num_return_sequences]

        output = item.copy()
        output['response'] = response
        output['steps_with_score'] = each_steps_with_score
        outputs.append(output)

    return outputs


def print_process_supervision(output):
    steps_with_score = output['steps_with_score']
    print('[Response] Start')
    for step_idx, step in enumerate(steps_with_score):
        print(
            f'[Steps-{step_idx}] Start\n'
            f"{step['step']}\n\n"
            f"{step['score']}\n"
            f"{step['num_mc_correct']}\n"
            f"{step['num_mc_total']}\n"
            f'[Steps-{step_idx}] End\n'
        )
    print('[Response] End')


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
            item2num[(str(item['image']), item['question_orig'])] += 1

    print(
        f'[{localtime()}] [Rank {torch.distributed.get_rank()}] '
        f'Begin to answer {len(dataloader)} batches '
        f'(about {len(dataloader) * args.batch_size} samples), '
        f'{args.prompt_path=}, '
        f'{results_file=}, '
        f'{min_len=}, '
        f'{len(item2num)=}'
    )

    log_freq = max(len(dataloader) // args.batch_size // 1000, 1)
    save_freq = max(len(dataloader) // args.batch_size // 25, 1)
    outputs = []
    for idx, (inputs, items) in enumerate(dataloader):
        assert len(inputs) == len(items)

        max_cnt = -1
        filtered_items = []
        filtered_inputs = []
        for i in range(len(inputs)):
            cnt = args.num_return_sequences - item2num[(str(items[i]['image']), items[i]['question'])]
            if cnt <= 0:
                continue
            max_cnt = max(max_cnt, cnt)
            filtered_items.append(items[i])
            filtered_inputs.append(inputs[i])

        items = filtered_items
        inputs = filtered_inputs
        if len(inputs) <= 0:
            print(
                f'[{localtime()}] '
                f'[Rank {torch.distributed.get_rank()}] '
                f'[Progress {idx}/{len(dataloader)}] '
                f'skip'
            )
            if idx % save_freq == 0 and idx < min_len:
                save_outputs(outputs, results_file)
                outputs = []
            continue

        curr_outputs = build_process_supervision(
            inputs=inputs,
            items=items,
            num_return_sequences=max_cnt,
        )
        assert len(curr_outputs) == len(inputs) * max_cnt

        for output_idx, output in enumerate(curr_outputs):
            output['num_mc_sequences'] = args.num_mc_sequences
            output['question_orig'] = output['question']
            output['question'] = inputs[output_idx // max_cnt][0].replace(IMAGE_TOKEN, IMG_PLACEHOLDER)
        outputs.extend(curr_outputs)

        if idx % log_freq == 0 and torch.distributed.get_rank() == 0:
            print(
                f'[Start]\n'
                f'[Prompt]\n{inputs[-1][0]}\n'
                f'[Image]\n{outputs[-1]["image"]}\n'
                f'[Question]\n{outputs[-1]["question_orig"]}\n'
                f'[Answer]\n{outputs[-1]["answer"]}\n'
                f'[End]\n'
            )
            print_process_supervision(outputs[-1])

        if idx % log_freq == 0:
            print(
                f'[{localtime()}] '
                f'[Rank {torch.distributed.get_rank()}] '
                f'[Progress {idx}/{len(dataloader)}] '
            )

        if idx % save_freq == 0 and idx < min_len:
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
    # lmdelpoy args
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--vit-batch-size', type=int, default=8)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--session-len', type=int, default=16384)
    parser.add_argument('--cache-max-entry-count', type=float, default=0.3)
    # generation args
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    # sampling args
    parser.add_argument('--num-return-sequences', type=int, default=4)
    parser.add_argument('--sample-start-idx', type=int, default=0)
    parser.add_argument('--sample-max-num', type=int, default=None)
    parser.add_argument('--prompt-version', type=str, default=None, choices=['en', 'en_v2', 'zh', 'zh_v2', 'en_r1', 'zh_r1'])
    parser.add_argument('--num-mc-sequences', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=12)
    parser.add_argument('--early-stop', action='store_true', default=True)
    parser.add_argument('--no-early-stop', action='store_false', dest='early_stop')
    args = parser.parse_args()
    args.tp = TP
    args.verification_mode = get_mode(os.path.basename(args.prompt_path))

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
        f'dynamic: {pipe.vl_encoder.model.config.dynamic_image_size}, '
        f'max_num: {pipe.vl_encoder.model.config.max_dynamic_patch}, '
        f'early_stop: {args.early_stop}, '
    )
    evaluate_chat_model()
