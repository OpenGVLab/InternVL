import base64

from PIL import Image
from typing import List, Tuple, Dict

from openai import OpenAI
from tools.internvlo1_pipeline.utils_eval import check_answer, parse_answer, fix_answer


# TODO: set correctness to 0 when all answer is distinct (备注：暂时不，否则root可能会直接mc_score=0受影响)
# TODO: update base_url
openai_client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
NODE_ID_GLOBAL = 0


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


class Node:
    def __init__(
        self,
        base_input: Tuple[str, Image.Image],
        answer_gt: str,
        answer_fix: bool,
        num_return_sequences: int,
        max_tiles: int,
        gen_config: Dict,
        parent=None,
        prefix: List[str] = None,
        use_advantage = False,
        prompt_version = 'en',
        min_token_threshold = 50,
    ):
        self.node_id = NODE_ID_GLOBAL

        global NODE_ID_GLOBAL
        NODE_ID_GLOBAL += 1

        self.prefix = prefix if prefix else []
        self.base_input = base_input
        self.answer_gt = answer_gt
        self.answer_fix = answer_fix
        self.num_return_sequences = num_return_sequences
        self.max_tiles = max_tiles
        self.gen_config = gen_config

        self.use_advantage = use_advantage
        self.prompt_version = prompt_version
        self.min_token_threshold = min_token_threshold

        self.parent = parent
        self.children: List[Node] = []
        self.is_visited = False

        self.rollouts: List[str] = []
        self.rollouts_is_visited: List[bool] = []
        self.correctness: List[float] = []

        self.selected_cnt = 0
        self.mc_estimation = None

    @staticmethod
    def split_response(response):
        return response.split(' ')

    @staticmethod
    def join_prefix(prefix):
        return ' '.join(prefix)

    def prepare_inputs(self):
        prompt, image = self.base_input
        image = base64.b64encode(image).decode('utf-8')
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'max_dynamic_patch': self.max_tiles, 'url': f'data:image/jpeg;base64,{image}'}},
            ]
        }]
        if self.prefix:
            messages.append({
                'role': 'prefix',
                'content': Node.join_prefix(self.prefix),
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

            if self.answer_fix:
                response = fix_answer(
                    response,
                    parse_answer(response, version=self.prompt_version),
                    self.answer_gt,
                )

            try:
                correctness = check_answer(parse_answer(response, version=self.prompt_version), self.answer_gt)
            except:
                correctness = 0.0
            self.correctness.append(correctness)

            # TODO: debug
            num_tokens = response.usage.completion_tokens_details.accepted_prediction_tokens

        self.mc_estimation = sum(self.correctness) / len(self.correctness)

    def update_cnt(self):
        self.selected_cnt += 1

    def binary_search(self, rollout_idx: int):
        assert 0 < self.mc_estimation < 1

        parent_node = self
        rollout = parent_node.rollouts[rollout_idx]
        correctness = parent_node.correctness[rollout_idx]
        rollout_words = Node.split_response(rollout)

        left = 0
        right = len(rollout_words) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if mid + 1 - left < parent_node.min_token_threshold:
                break

            current_prefix = parent_node.prefix + rollout_words[left:mid+1]
            current_suffix = Node.join_prefix(rollout_words[mid+1:])
            node = Node(
                base_input=parent_node.base_input,
                answer_gt=parent_node.answer_gt,
                num_return_sequences=parent_node.num_return_sequences,
                gen_config=parent_node.gen_config,
                parent=parent_node,
                prefix=current_prefix,
                use_advantage=parent_node.use_advantage,
                prompt_version=parent_node.prompt_version,
                min_token_threshold=parent_node.min_token_threshold,
            )
            node.register_rollout(current_suffix, correctness)
            node.update_rollouts()
            NODE_ID_GLOBAL += 1

            if node.mc_estimation > 0:
                left = mid + 1
                parent_node.children.append(node)
                parent_node = node

                if node.mc_estimation >= 1:
                    break
            else:
                right = mid - 1
                parent_node.children.append(node)

        # If all nodes satisfy mc_score > 0
        return

    @property
    def weight(self):
        if self.use_advantage:
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
        answer_fix: bool,
        alpha: float,
        beta: float,
        base_length: float,
        c_puct: float,
        num_return_sequences: int,
        max_tiles: int,
        use_advantage: bool,
        prompt_version: str,
        min_token_threshold: int,
    ):
        self.base_input = base_input
        self.answer_gt = answer_gt
        self.answer_fix = answer_fix
        self.alpha = alpha
        self.beta = beta
        self.base_length = base_length
        self.c_puct = c_puct
        self.num_return_sequences = num_return_sequences
        self.max_tiles = max_tiles

        self.use_advantage = use_advantage
        self.prompt_version = prompt_version
        self.min_token_threshold = min_token_threshold

        self.root = Node(
            base_input=self.base_input,
            answer_gt=self.answer_gt,
            answer_fix=self.answer_fix,
            num_return_sequences=self.num_return_sequences,
            max_tiles=self.max_tiles,
            gen_config=self.gen_config,
            parent=None,
            prefix=[],
            use_advantage=self.use_advantage,
            prompt_version=self.prompt_version,
            min_token_threshold=self.min_token_threshold,
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
        while len(nodes_list) > 0:
            node = nodes_list.pop(0)
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

    @staticmethod
    def remove_unnecessary_attributes_obj(obj):
        del obj.base_input
        del obj.num_return_sequences
        del obj.max_tiles
        # del obj.use_advantage
        # del obj.prompt_version
        del obj.min_token_threshold

    def remove_unnecessary_attributes(self):
        del self.alpha
        del self.beta
        del self.base_length
        del self.c_puct
        del self.available_nodes

        Tree.remove_unnecessary_attributes_obj(self)

        nodes_list = [self.root]
        while len(nodes_list) > 0:
            node = nodes_list.pop(0)
            nodes_list.extend([child for child in node.children if not child.is_visited])

            del node.answer_gt
            Tree.remove_unnecessary_attributes_obj(node)


def build_trees(args, inputs, items, gen_config):
    # initialize rollouts of root node
    tree = Tree(
        base_input=inputs[0],
        answer_gt=items[0]['answer'],
        answer_fix=args.answer_fix,
        alpha=args.alpha,
        beta=args.beta,
        base_length=args.base_length,
        c_puct=args.c_puct,
        num_return_sequences=args.num_return_sequences,
        max_tiles=args.max_num,
        gen_config=gen_config,
        use_advantage=args.use_advantage,
        prompt_version=args.prompt_version,
        min_token_threshold=args.min_token_threshold,
    )

    while tree.num_nodes <= args.max_nodes and tree.has_available_nodes():
        # select nodes
        node, rollout_idx = tree.select_node()
        # binary search (sample rollouts, extend new nodes)
        node.update_cnt()
        node.binary_search(rollout_idx)
        # main the tree info
        tree.maintain(node)

    tree.remove_unnecessary_attributes()

    item = items[0].copy()
    item['tree'] = tree

    return [item]


def print_trees(tree: Tree):
    answer_gt = tree.answer_gt
    print(f'Global Info: {answer_gt=}')

    num_nodes = 0
    nodes_list: List[Node] = [(tree.root, 0)]
    while len(nodes_list) > 0:
        node, depth = nodes_list.pop(-1)
        nodes_list.extend(reversed([
            (child, depth+1)
            for child in node.children if not child.is_visited
        ]))

        node_id = node.node_id
        prefix = Node.join_prefix(node.prefix)
        rollouts = node.rollouts
        correctness = node.correctness
        mc_score = node.mc_estimation
        children = node.children

        sep = '\t' * depth
        node_info = [
            f'{sep}[Node {node_id}] {prefix=} {len(children)=} {mc_score=}',
            f'{sep}\t[Rollouts]',
        ]
        for rollout, is_correct in zip(rollouts, correctness):
            node_info.append(f'{sep}\t{rollout} ({is_correct=})')

        print('\n'.join(node_info))
    print(f'[Finish] {num_nodes=}, {tree.num_nodes=}')
