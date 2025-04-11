import json
import os
from argparse import ArgumentParser
from collections import defaultdict

from tools.reasoning_data_pipeline.utils.constants import PRM_SYSTEM_PROMPT
from tools.reasoning_data_pipeline.utils.utils import load_outputs


def save_outputs(outputs, results_file):
    outputs = sorted(outputs, key=lambda x:str(x['image']))

    with open(results_file, 'w') as file:
        for output in outputs:
            file.write(json.dumps(output) + '\n')

    print(f'Results ({len(outputs)=}) saved to {results_file}')


def item2conv_prm(item):
    image = item['image']
    question = item['question']
    steps_with_score = item['steps_with_score']

    threshold = args.mc_threshold
    conversations = [{'from': 'system', 'value': PRM_SYSTEM_PROMPT}]
    for step_idx, step in enumerate(steps_with_score):
        query = step['step']
        if step_idx == 0:
            query = f'### Question:\n{question}\n\n### Solution Process:\n{query}'

        conversations.extend([
            {'from': 'human', 'value': query},
            {'from': 'gpt', 'value': '+' if step['score'] > threshold else '-'},
        ])

        if args.early_stop and step['score'] <= threshold:
            break

    return {
        'id': -1,
        'image': image,
        'conversations': conversations,
    }


def item2conv_orm(item):
    image = item['image']
    question = item['question']
    steps_with_score = item['steps_with_score']

    if 'response' in item:
        response = item['response']
    else:
        response = '\n\n'.join([step['step'] for step in steps_with_score]).strip()

    query = f'### Question:\n{question}\n\n### Solution Process:\n{response}'
    last_step_score = steps_with_score[-1]['score']

    threshold = args.mc_threshold
    conversations = [
        {'from': 'system', 'value': PRM_SYSTEM_PROMPT},
        {'from': 'human', 'value': query},
        {'from': 'gpt', 'value': '+' if last_step_score > threshold else '-'},
    ]

    return {
        'id': -1,
        'image': image,
        'conversations': conversations,
    }


def main():
    if not os.path.exists(args.data_dir):
        print(f'Dir does not exist: {args.data_dir}')
        exit(0)

    for filename in os.listdir(args.data_dir):
        if not filename.endswith('.jsonl'):
            continue

        save_dir = args.save_dir
        ds_name = os.path.basename(filename).replace('.jsonl', '')
        os.makedirs(os.path.join(save_dir, 'raw'), exist_ok=True)

        pairs_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_prm.jsonl')
        pairs_orm_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_orm.jsonl')

        if os.path.exists(pairs_save_path) and not args.overwrite:
            continue

        info = defaultdict(int)
        id2scores = defaultdict(list)
        statistics = defaultdict(list)

        convs_prm = []
        convs_orm = []
        items = load_outputs(os.path.join(args.data_dir, filename))

        for item in items:
            image = item['image']
            question = item['question']
            steps_with_score = item['steps_with_score']

            score = steps_with_score[-1]['score']
            id2scores[(str(image), question)].append(score)

        for item in items:
            convs_prm.append(item2conv_prm(item))
            convs_orm.append(item2conv_orm(item))
            statistics['num_turns'].append(len(convs_prm[-1]['conversations']))

        print(f'[{filename}]')
        for k, v in info.items():
            print(k, v)
        for k, v in statistics.items():
            print(f'{k}: max={max(v)}, min={min(v)}, mean={sum(v) / len(v)}, total={sum(v)}')
        print()

        save_outputs(convs_prm, pairs_save_path)
        if args.include_orm_data:
            save_outputs(convs_orm, pairs_orm_save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='')
    parser.add_argument('--mc-threshold', type=float, default=0.0)
    parser.add_argument('--early-stop', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--include-orm-data', action='store_true', default=False)
    args = parser.parse_args()

    main()
