import io
import os
import json
import argparse
import torch

from PIL import Image
from collections import defaultdict
# from lmdeploy import GenerationConfig, TurbomindEngineConfig, VisionConfig, pipeline
# from lmdeploy.model import InternVL2InternLM2, Qwen7BChat
from lmdeploy.vl.constants import IMAGE_TOKEN

from tools.internvlo1_pipeline.utils_eval import get_mode
from tools.internvlo1_pipeline.utils_mcts import build_trees, print_trees, model_name
from tools.internvlo1_pipeline.utils_dist import (
    init_dist,
    localtime,
    get_global_min,
    InferenceSampler,
    multimodal_collate_fn as collate_fn,
    save_outputs_with_pickle as save_outputs,
    load_outputs_with_pickle as load_outputs,
)

try:
    from petrel_client.client import Client
    client = Client()
except:
    import warnings
    warnings.warn(
        'Fail to import petrel_client! '
        'You can ignore this warning if you do not need to load image from ceph.'
    )


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

IMG_PLACEHOLDER = '<image>'
INSTRUCTION_EN = (
    'Your task is to answer the question below. '
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    '\n\n'
    'Question:'
    '\n\n'
    '{question}'
)
INSTRUCTION_ZH = (
    "你的任务是回答以下问题。在回答之前，请逐步推理说明您的思路。当你准备好给出答案时，请使用以下格式：\"答案: ...\""
    '\n\n'
    '问题:'
    '\n\n'
    '{question}'
)
VALID_INSTRUCTIONS = [
    'Answer the question using a single word or phrase.',
    "Answer with the option's letter from the given choices directly.",
    'Please answer Yes or No.',
]
VALID_INSTRUCTIONS = set(VALID_INSTRUCTIONS)


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
            step = len(self.data) // sample_max_num
            self.data = self.data[::step]

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
                image = client.get(image)
            else:
                with open(image, 'rb') as image_file:
                    image = image_file.read()
            images_new.append(image)
        images = images_new

        for instruction in VALID_INSTRUCTIONS:
            if question.endswith(instruction):
                question = question[:-len(instruction)].strip()
        question = INSTRUCTION.format(question=question)

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

    gen_config = dict(
        top_p=args.top_p,
        temperature=args.temperature,
    )

    item2num = defaultdict(int)
    results_file = os.path.basename(args.prompt_path)
    results_file = os.path.join(args.out_dir, results_file)
    results_file = results_file.replace('.jsonl', '.pkl')
    if os.path.exists(results_file):
        items = load_outputs(results_file)
        for item in items:
            item2num[(str(item['image']), item['question_orig'])] += 1
        del items

    print(
        f'[{localtime()}] [Rank {torch.distributed.get_rank()}] '
        f'Begin to answer {len(dataloader)} batches '
        f'(about {len(dataloader) * args.batch_size} samples), '
        f'{args.prompt_path=}, '
        f'{len(item2num)=}'
    )

    log_freq = max(len(dataloader) // args.batch_size // 1000, 1)
    outputs = []
    for idx, (inputs, items) in enumerate(dataloader):
        assert len(inputs) == len(items)

        filtered_items = []
        filtered_inputs = []
        for i in range(len(inputs)):
            cnt = item2num[(str(items[i]['image']), items[i]['question'])]
            if cnt > 0:
                continue
            filtered_items.append(items[i])
            filtered_inputs.append(inputs[i])

        items = filtered_items
        inputs = filtered_inputs
        if len(inputs) <= 0:
            continue

        outputs.extend(build_trees(args=args, inputs=inputs, items=items, gen_config=gen_config))

        for output in outputs:
            output['question_orig'] = output['question']
            output['question'] = inputs[0][0].replace(IMAGE_TOKEN, IMG_PLACEHOLDER)

        if idx % log_freq == 0 and torch.distributed.get_rank() == 0:
            print(
                f'[Start]\n'
                f'[Prompt]\n{inputs[-1][0]}\n'
                f'[Image]\n{outputs[-1]["image"]}\n'
                f'[Question]\n{outputs[-1]["question_orig"]}\n'
                f'[Answer]\n{outputs[-1]["answer"]}\n'
                f'[End]\n'
            )
            print_trees(outputs[-1]['tree'])

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
    parser.add_argument('--prompt-path', type=str, default='outputs/correctness_prompt_mmpr/m3cot_train_extracted.jsonl')
    parser.add_argument('--out-dir', type=str, default='outputs/prm_mmpr')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--sample-max-num', type=int, default=None)
    parser.add_argument('--prompt-version', type=str, default='en', choices=['en', 'zh'])
    # hyper-parameters for mcts
    parser.add_argument('--num-return-sequences', type=int, default=4)
    parser.add_argument('--max-nodes', type=int, default=32)
    parser.add_argument('--min-token-threshold', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--base-length', type=float, default=500)
    parser.add_argument('--c-puct', type=float, default=0.125)
    parser.add_argument('--use-advantage', action='store_true')
    parser.add_argument('--answer-fix', action='store_true', default=True)
    args = parser.parse_args()
    args.tp = 1
    args.verification_mode = get_mode(os.path.basename(args.prompt_path))

    global INSTRUCTION
    if args.prompt_version == 'zh':
        INSTRUCTION = INSTRUCTION_ZH
    elif args.prompt_version == 'en':
        INSTRUCTION = INSTRUCTION_EN
    else:
        raise NotImplementedError(f'Unsupported prompt version {args.prompt_version}')

    assert args.temperature > 0
    assert args.batch_size == 1, 'only batch_size=1 is supported'

    init_dist(args)

    model_name = '_'.join(model_name.split('/')[-2:])
    args.out_dir = os.path.join(args.out_dir, model_name, f'max_tiles_{args.max_num}')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Begin to sample data from model {model_name}')
    evaluate_chat_model()
