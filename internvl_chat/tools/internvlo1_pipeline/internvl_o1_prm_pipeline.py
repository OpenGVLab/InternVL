import io
import os
import json
import copy
import argparse
import torch

from PIL import Image
from collections import defaultdict
from typing import List, Tuple, Dict
from openai import OpenAI
from lmdeploy import GenerationConfig, TurbomindEngineConfig, VisionConfig, pipeline
from lmdeploy.model import InternVL2InternLM2, Qwen7BChat
from lmdeploy.vl.constants import IMAGE_TOKEN

from tools.internvlo1_pipeline.utils_eval import check_answer, parse_answer
from tools.internvlo1_pipeline.utils_dist import (
    InferenceSampler,
    multimodal_collate_fn as collate_fn,
    init_dist, save_outputs, localtime, get_global_min,
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

# TODO: set correctness to 0 when all answer is distinct (备注：暂时不，否则root可能会直接mc_score=0受影响)
# TODO: update base_url
openai_client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')

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


# TODO: update the source code when deploying API
def messages2prompt(self, messages, sequence_start=True, **kwargs):
    """Return the prompt that is concatenated with other elements in the
    chat template.

    Args:
        messages (str | List): user's input prompt
    Returns:
        str: the concatenated prompt
    """
    if isinstance(messages, str):
        return self.get_prompt(messages, sequence_start)

    prefix_info = None
    if messages[-1]['role'] == 'prefix':
        prefix_info = messages.pop(-1)
        prefix_info = prefix_info['content']

    box_map = dict(user=self.user,
                    assistant=self.assistant,
                    system=self.system)
    eox_map = dict(user=self.eoh,
                    assistant=self.eoa + self.separator,
                    system=self.eosys)
    ret = ''
    if self.meta_instruction is not None and sequence_start:
        if len(messages) and messages[0]['role'] != 'system':
            ret += f'{self.system}{self.meta_instruction}{self.eosys}'
    for message in messages:
        role = message['role']
        content = message['content']
        ret += f'{box_map[role]}{content}{eox_map[role]}'
    ret += f'{self.assistant}'

    if prefix_info is not None:
        ret += f'{prefix_info}'

    return ret


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
                image = io.BytesIO(client.get(image))
            image = Image.open(image).convert('RGB')
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


class Node:
    def __init__(
        self,
        node_id: int,
        base_input: Tuple[str, Image.Image],
        answer_gt: str,
        num_return_sequences: int,
        gen_config: Dict,
        parent=None,
        prefix: List[str] = None,
    ):
        self.node_id = node_id
        self.prefix = prefix if prefix else []
        self.base_input = base_input
        self.answer_gt = answer_gt
        self.num_return_sequences = num_return_sequences
        self.gen_config = gen_config

        self.parent = parent
        self.children: List[Node] = []
        self.is_visited = False

        self.rollouts: List[str] = []
        self.rollouts_is_visited: List[bool] = []
        self.correctness: List[float] = []

        self.selected_cnt = 0
        self.mc_estimation = None

    def split_response(self, response):
        return response.split()

    def join_prefix(self):
        return ' '.join(self.prefix)

    def prepare_inputs(self):
        prompt, image = self.base_input
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_data', 'data': image},
            ]
        }]
        if self.prefix:
            messages.append({
                'role': 'prefix',
                'content': self.join_prefix(),
            })

        return messages

    def update_visible(self):
        self.is_visited = True

    def update_rollout_visible(self, rollout_idx: int):
        self.rollouts_is_visited[rollout_idx] = True

    def register_rollout(self, rollout, correctness):
        self.rollouts.append(rollout)
        self.rollouts_is_visited.append(True)
        self.correctness.append(correctness)

    def update_rollouts(self):
        messages = self.prepare_inputs()
        for _ in range(self.num_return_sequences - len(self.rollouts)):
            response = openai_client.chat.completions.create(messages=messages, **self.gen_config)
            response = response.choices[0].message.content
            self.rollouts.append(response)
            self.rollouts_is_visited.append(False)

            try:
                correctness = check_answer(parse_answer(response, version=args.prompt_version), self.answer_gt)
            except:
                correctness = 0
            self.correctness.append(correctness)

            # TODO: debug
            num_tokens = response.usage.completion_tokens_details.accepted_prediction_tokens

        self.mc_estimation = sum(self.correctness) / len(self.correctness)

    def update_cnt(self):
        self.selected_cnt += 1

    def binary_search(self, rollout_idx: int):
        assert 0 < self.mc_estimation < 1

        rollout = self.rollouts[rollout_idx]
        correctness = self.correctness[rollout_idx]
        rollout_words = self.split_response(rollout)

        left = 0
        right = len(rollout_words) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if mid + 1 - left < args.threshold:
                break

            current_prefix = self.prefix + rollout_words[left:mid+1]
            current_suffix = self.join_prefix(rollout_words[mid+1:])
            node = Node(
                node_id=self.node_id + len(self.children) + 1,
                base_input=self.base_input,
                answer_gt=self.answer_gt,
                num_return_sequences=self.num_return_sequences,
                gen_config=self.gen_config,
                parent=self,
                prefix=current_prefix,
            )
            node.register_rollout(current_suffix, correctness)
            node.update_rollouts()

            if node.mc_estimation > 0:
                left = mid + 1
                self.children.append(node)

                if node.mc_estimation >= 1:
                    break
            else:
                right = mid - 1
                self.children.append(node)

        # If all nodes satisfy mc_score > 0
        return

    @property
    def weight(self):
        if args.use_advantage:
            return self.mc_estimation - self.parent.mc_estimation
        return self.mc_estimation

    @property
    def is_leaf(self):
        return len(self.children) == 0


class Tree:
    def __init__(
        self,
        base_input: Tuple[str, Image.Image],
        answer_gt: str,
        alpha: float,
        beta: float,
        base_length: float,
        c_puct: float,
        num_return_sequences: int,
    ):
        self.base_input = base_input
        self.alpha = alpha
        self.beta = beta
        self.base_length = base_length
        self.c_puct = c_puct
        self.num_return_sequences = num_return_sequences

        self.root = Node(
            node_id=0,
            base_input=self.base_input,
            answer_gt=answer_gt,
            num_return_sequences=self.num_return_sequences,
            gen_config=self.gen_config,
            parent=None,
            prefix=[],
        )
        self.root.update_rollouts()
        self.root.update_visible()

        self.num_nodes = 1
        self.total_cnt = 0
        self.available_nodes: List[Tuple[Node, int]] = []

        if 0 < self.root.mc_estimation < 1:
            for rollout_idx in range(len(self.root.rollouts)):
                self.root.update_rollout_visible(rollout_idx)
                self.available_nodes.append((self.root, rollout_idx))

    def get_weight(self, node, rollout_idx):
        weight_q = (
            self.alpha ** (1 - node.weight) *
            self.beta ** (len(node.rollouts[rollout_idx]) / self.base_length)
        )

        weight_u = (
            self.total_cnt ** 0.5 /
            (1 + node.selected_cnt)
        )

        weight = weight_q + weight_u
        return weight

    def select_node(self):
        selected_node, selected_edge_idx = self.available_nodes[0]
        max_idx = 0
        max_weight = self.get_weight(selected_node, selected_edge_idx)

        for i in range(1, len(self.available_nodes)):
            node, rollout_idx = self.available_nodes[i]
            weight = self.get_weight(node, rollout_idx)

            if weight > max_weight:
                max_idx = i
                max_weight = weight
                selected_node = node
                selected_edge_idx = rollout_idx

        self.available_nodes.pop(max_idx)
        return selected_node, selected_edge_idx

    def maintain(self, node):
        nodes_list = [child for child in node.children if not child.is_visited]
        for node in nodes_list:
            if node.is_visited:
                continue

            node.update_visible()
            nodes_list.extend([child for child in node.children if not child.is_visited])

            if 0 < node.mc_estimation < 1:
                for rollout_idx in range(len(node.rollouts)):
                    if not node.rollouts_is_visited[rollout_idx]:
                        node.update_rollout_visible(rollout_idx)
                        self.available_nodes.append((node, rollout_idx))

            self.num_nodes += 1
        self.total_cnt += 1

    @property
    def has_available_nodes(self):
        return len(self.available_nodes) > 0


def build_trees(inputs, items, gen_config):
    # initialize rollouts of root node
    tree = Tree(
        base_input=inputs[0],
        answer_gt=items[0]['answer'],
        alpha=args.alpha,
        beta=args.beta,
        base_length=args.base_length,
        c_puct=args.c_puct,
        num_return_sequences=args.num_return_sequences,
        gen_config=gen_config,
    )

    while tree.num_nodes <= args.max_nodes and tree.has_available_nodes():
        # select nodes
        node, rollout_idx = tree.select_node()
        # binary search (sample rollouts, extend new nodes)
        node.update_cnt()
        node.binary_search(rollout_idx)
        # main the tree info
        tree.maintain(node)

    # TODO: how to save the tree?
    item = items[0].copy()
    item['tree'] = tree

    return [item]


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
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
    )

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
        f'{len(item2num)=}'
    )

    log_freq = max(len(dataloader) // args.batch_size // 100, 1)
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

        outputs.extend(build_trees(inputs=inputs, items=items, gen_config=gen_config))

        if idx % log_freq == 0 and torch.distributed.get_rank() == 0:
            print(
                f'[Prompt]\n{inputs[-1][0]}\n'
                f'[Image]\n{outputs[-1]["image"]}\n'
                f'[Question]\n{outputs[-1]["question_orig"]}\n'
                f'[Output]\n{outputs[-1]["tree"].root.rollouts[0]}\n'
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
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--prompt-path', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='sampled_outputs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--vit-batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--min-new-tokens', type=int, default=1)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--sample-max-num', type=int, default=None)
    parser.add_argument('--prompt-version', type=str, default='en', choices=['en', 'zh'])
    # hyper-parameters for mcts
    parser.add_argument('--num-return-sequences', type=int, default=4)
    parser.add_argument('--max-nodes', type=int, default=96)
    parser.add_argument('--threshold', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--base-length', type=float, default=500)
    parser.add_argument('--c-puct', type=float, default=0.125)
    parser.add_argument('--use-advantage', action='store_true')
    args = parser.parse_args()

    global INSTRUCTION
    if args.prompt_version == 'zh':
        INSTRUCTION = INSTRUCTION_ZH
    elif args.prompt_version == 'en':
        INSTRUCTION = INSTRUCTION_EN
    else:
        raise NotImplementedError(f'Unsupported prompt version {args.prompt_version}')

    assert args.temperature > 0
    assert args.batch_size == 1, 'only batch_size=1 is supported'
    assert args.tp == 1, 'model is invoked in the format of API'

    init_dist(args)

    model_name = '_'.join(args.checkpoint.split('/')[-2:])
    args.out_dir = os.path.join(args.out_dir, model_name, f'max_tiles_{args.max_num}')
    os.makedirs(args.out_dir, exist_ok=True)

    # Qwen7BChat.messages2prompt = messages2prompt
    # InternVL2InternLM2.messages2prompt = messages2prompt

    # vision_config = VisionConfig(max_batch_size=args.vit_batch_size)
    # pipe = pipeline(
    #     args.checkpoint,
    #     vision_config=vision_config,
    #     backend_config=TurbomindEngineConfig(session_len=8192, cache_max_entry_count=0.1, tp=args.tp)
    # )
    # pipe.vl_encoder.model.config.max_dynamic_patch = args.max_num
    # pipe.vl_encoder.model.config.dynamic_image_size = args.dynamic

    # # lmdeploy will update the current_device
    # torch.cuda.set_device(int(os.environ['RANK']) % torch.cuda.device_count())

    print(
        f'Begin to sample data from model {args.checkpoint}, '
        # f'dynamic: {pipe.vl_encoder.model.config.dynamic_image_size}, '
        # f'max_num: {pipe.vl_encoder.model.config.max_dynamic_patch}, '
    )
    evaluate_chat_model()
