import os
import json
import random

from typing import List, Dict
from argparse import ArgumentParser
from collections import defaultdict
from tools.internvlo1_pipeline.utils_mcts import Tree, Node, load_outputs_with_pickle


def save_outputs(outputs, results_file):
    outputs = sorted(outputs, key=lambda x:x['image'])

    with open(results_file, 'a') as file:
        for output in outputs:
            file.write(json.dumps(output) + '\n')

    print(f'Results ({len(outputs)=}) saved to {results_file}')


# {'response': 'xxx', 'mc_score': xxx}
def tree2list(item) -> List[Dict]:
    tree: Tree = item['tree']
    response_list = []

    nodes_list: List[Node] = [tree.root]
    while len(nodes_list) > 0:
        node = nodes_list.pop(-1)
        nodes_list.extend([child for child in node.children if not child.is_visited])

        prefix = Node.join_prefix(node.prefix)
        mc_score = node.mc_estimation

        response_list.append({
            'response': prefix,
            'mc_score': mc_score,
            'num_words': len(node.prefix),
        })

    return response_list


# {'image': 'xxx', 'question': 'xxx', 'chosen': 'xxx', 'chosen_mc_score': xxx, 'rejected': 'xxx', 'rejected_mc_score': xxx}
def list2pair(item):
    pairs = []
    image = item['image']
    question = item['question']
    response_list = item['response_list']
    response_list = sorted(response_list, key=lambda x:x['num_words'])

    for i in range(0, len(response_list), 2):
        first = response_list[i]
        second = response_list[i+1]

        if first['mc_score'] > second['mc_score']:
            chosen = first
            rejected = second
        elif first['mc_score'] < second['mc_score']:
            chosen = first
            rejected = second
        else:
            continue

        pairs.append({
            'image': image,
            'question': question,
            'chosen': chosen['response'],
            'chosen_mc_score': chosen['mc_score'],
            'rejected': rejected['response'],
            'rejected_mc_score': rejected['mc_score'],
        })

    if args.num_pairs_per_tree > 0:
        return random.sample(pairs, args.num_pairs_per_tree)
    return pairs


def main():
    for filename in os.listdir(args.data_dir):
        if not filename.endswith('.jsonl'):
            continue

        save_dir = args.save_dir
        ds_name = os.path.basename(filename).replace('.jsonl', '')
        os.makedirs(os.path.join(save_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'lines'), exist_ok=True)

        pairs_save_path = os.path.join(save_dir, 'raw', f'{ds_name}.jsonl')
        lines_save_path = os.path.join(save_dir, 'lines', f'{ds_name}.jsonl')

        statistics = defaultdict(list)
        pairs = []
        items = load_outputs_with_pickle(os.path.join(args.data_dir, filename))
        for item in items:
            item['response_list'] = tree2list(item)
            item.pop('tree')

            local_pairs = list2pair(item)
            pairs.extend(local_pairs)

            statistics['num_pairs'].append(len(local_pairs))
            statistics['num_responses'].append(len(item['response_list']))

        print(f'[{filename}]')
        for k, v in statistics:
            print(f'\t{k}: max={max(v)}, min={min(v)}, mean={sum(v) / len(v)}, total={sum(v)}')
        print()

        save_outputs(pairs, pairs_save_path)
        save_outputs(items, lines_save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='')
    parser.add_argument('--num-pairs-per-tree', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()

    main()
