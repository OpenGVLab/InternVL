import argparse
import json
import os
import random
from collections import defaultdict

from tools.reasoning_data_pipeline.utils.accuracy_reward import (check_answer,
                                                                 fix_answer,
                                                                 get_mode,
                                                                 parse_answer)
from tqdm import tqdm

random.seed(0)


# you can download this file from this url: https://huggingface.co/datasets/Weiyun1025/M3CoT-ScienceQA-Format/blob/main/train_pair_with_res.jsonl
gt_path_map = {
    'm3cot': 'M3CoT/train_pair_with_res.jsonl',
}


def _build_items_based_on_correctness(lines, mode):
    pos_id2item = defaultdict(list)
    neg_id2item = defaultdict(list)
    neg_format_id2item = defaultdict(list)
    for line in tqdm(lines, desc='check_answer'):
        item = json.loads(line)
        image = str(item.get('image', ''))
        question = item['question']
        answer_gt = item['answer']
        response = item['response']
        prompt_version = item.get('prompt_version', 'en')

        if args.use_correctness_cache:
            correct = int(item['is_correct'])
        else:
            try:
                _, answer_pred = parse_answer(response, prompt_version=prompt_version)
                item['answer_pred'] = answer_pred
            except:
                print('Fail to parse answer from response:', response)
                item['answer_pred'] = 'None'
                neg_format_id2item[(image, question, answer_gt)].append(item)
                continue

            if args.answer_fix and prompt_version in ['en', 'zh']:
                try:
                    item['response'] = fix_answer(response, answer_pred, answer_gt)
                except:
                    item['answer_pred'] = 'None'
                    neg_format_id2item[(image, question, answer_gt)].append(item)
                    continue

                response = item['response']
                answer_pred = item['answer_pred']

            correct = check_answer(answer_pred, answer_gt, mode=mode)

        assert correct in [0, 1], correct

        if correct:
            pos_id2item[(image, question, answer_gt)].append(item)
        else:
            neg_id2item[(image, question, answer_gt)].append(item)

    return pos_id2item, neg_id2item, neg_format_id2item


def build_neg_based_on_correctness(lines, mode):
    pos_id2item, neg_id2item, neg_format_id2item = _build_items_based_on_correctness(lines, mode=mode)

    all_correct = 0
    for key in pos_id2item:
        if key not in neg_id2item:
            all_correct += 1

    all_incorrect_keys = []
    all_incorrect = 0
    for key in neg_id2item:
        if key not in pos_id2item:
            all_incorrect += 1
            all_incorrect_keys.append(key)

    print(
        f'[build_neg_based_on_correctness] '
        f'num_pos_samples={sum(len(v) for v in pos_id2item.values())}, '
        f'num_neg_samples={sum(len(v) for v in neg_id2item.values())}, '
        f'num_format_neg_samples={sum(len(v) for v in neg_format_id2item.values())}, '
        f'{all_correct=}, '
        f'{all_incorrect=}, '
    )
    return pos_id2item, neg_id2item, neg_format_id2item, all_incorrect_keys


def _build_pair_based_on_pos_neg(item_pos, item_neg):
    image_pos = item_pos.get('image', '')
    question_pos = item_pos['question']
    answer_gt_pos = item_pos['answer']
    response_pos = item_pos['response']

    image_neg = item_neg.get('image', '')
    question_neg = item_neg['question']
    answer_gt_neg = item_neg['answer']
    response_neg = item_neg['response']

    assert (image_pos, question_pos, answer_gt_pos) == (image_neg, question_neg, answer_gt_neg)

    pair = {
        'image': image_pos,
        'question': question_pos,
        'chosen': response_pos,
        'rejected': response_neg,
        'answer_gt': answer_gt_pos,
    }
    if 'meta' in item_pos:
        meta_pos = item_pos['meta']
        pair['chosen_meta'] = meta_pos
    if 'meta' in item_neg:
        meta_neg = item_neg['meta']
        pair['rejected_meta'] = meta_neg
    return pair


def build_pairs_based_on_pos_neg(pos_id2item, neg_id2item, allow_entailment=False):
    info = defaultdict(int)
    pair_samples = []
    for key in pos_id2item:
        if key not in neg_id2item:
            continue

        curr_pair_samples = []
        for item_pos in pos_id2item[key]:
            for item_neg in neg_id2item[key]:

                if not allow_entailment and item_pos['answer_pred'].lower() in item_neg['answer_pred'].lower():
                    info['entail_skip'] += 1
                    continue

                curr_pair_samples.append(_build_pair_based_on_pos_neg(item_pos=item_pos, item_neg=item_neg))

        if len(curr_pair_samples) == 0:
            info['key_without_pairs'] += 1

        info['max_possible_pairs'] += len(pos_id2item[key]) * len(neg_id2item[key])
        pair_samples.extend(random.sample(curr_pair_samples, min(len(curr_pair_samples), NUM_PAIRS_PER_KEY)))

    only_in_pos = len(pos_id2item.keys() - neg_id2item.keys())
    only_in_neg = len(neg_id2item.keys() - pos_id2item.keys())
    in_both = len(pos_id2item.keys() & neg_id2item.keys())

    info_str = ', '.join([f'{k}={v}' for k, v in info.items()])
    print(
        f'[build_pairs_based_on_pos_neg {NUM_PAIRS_PER_KEY=}] '
        f'num_pairs={len(pair_samples)}, '
        f'{only_in_pos=}, '
        f'{only_in_neg=}, '
        f'{in_both=}, '
        f'{info_str}, '
    )
    return pair_samples


def save_items(items, save_path, question_only=False, all_incorrect_keys=None):
    if question_only:
        items_set = set()
        keys = all_incorrect_keys if all_incorrect_keys is not None else items.keys()
        for key in keys:
            values = items[key]
            for item in values:
                items_set.add((str(item.get('image', '')), item['question'], item['answer']))

        items_list = []
        for item in items_set:
            try:
                image = eval(item[0])
            except:
                image = item[0]

            if image:
                items_list.append({
                    'image': image,
                    'question': item[1],
                    'answer': item[2],
                })
            else:
                items_list.append({
                    'question': item[1],
                    'answer': item[2],
                })

        with open(save_path, 'w') as file:
            for item in items_list:
                file.write(json.dumps(item) + '\n')
    else:
        with open(save_path, 'w') as file:
            for values in items.values():
                for item in values:
                    file.write(json.dumps(item) + '\n')


def save_pairs(pairs, save_path):
    distinct_pairs = set()
    chosen_meta_dict = {}
    rejected_meta_dict = {}
    for pair in pairs:
        pair = pair.copy()
        image = str(pair['image'])
        question = pair['question']
        chosen = pair['chosen']
        rejected = pair['rejected']
        answer_gt = pair['answer_gt']

        if 'chosen_meta' in pair:
            choosen_meta = pair.pop('chosen_meta')
            chosen_meta_dict[(image, question, chosen, rejected, answer_gt)] = choosen_meta

        if 'rejected_meta' in pair:
            rejected_meta = pair.pop('rejected_meta')
            rejected_meta_dict[(image, question, chosen, rejected, answer_gt)] = rejected_meta

        distinct_pairs.add((image, question, chosen, rejected, answer_gt))

        assert pair.keys() == {'image', 'question', 'chosen', 'rejected', 'answer_gt'}, pair.keys()

    filtered_pairs = []
    for pair in distinct_pairs:
        image, question, chosen, rejected, answer_gt = pair
        try:
            image = eval(image)
        except:
            pass

        if image:
            filtered_pair = {
                'image': image,
                'question': question,
                'chosen': chosen,
                'rejected': rejected,
                'answer_gt': answer_gt,
            }
        else:
            filtered_pair = {
                'question': question,
                'chosen': chosen,
                'rejected': rejected,
                'answer_gt': answer_gt,
            }

        image = str(image)
        if (image, question, chosen, rejected, answer_gt) in chosen_meta_dict:
            filtered_pair['chosen_meta'] = chosen_meta_dict[(image, question, chosen, rejected, answer_gt)]

        if (image, question, chosen, rejected, answer_gt) in rejected_meta_dict:
            filtered_pair['rejected_meta'] = rejected_meta_dict[(image, question, chosen, rejected, answer_gt)]

        filtered_pairs.append(filtered_pair)

    if len(filtered_pairs) == 0:
        return

    with open(save_path, 'w') as file:
        for pair in filtered_pairs:
            file.write(json.dumps(pair) + '\n')
    print(f'Save {len(pairs)} pairs ({len(filtered_pairs)} distinct pairs) in {save_path}')


def main(args):
    if not os.path.exists(args.data_dir):
        print(f'{args.data_dir} do not exist, skip process')
        return

    for filename in os.listdir(args.data_dir):
        if not filename.endswith('.jsonl'):
            continue

        save_dir = args.save_dir
        ds_name = os.path.basename(filename).replace('.jsonl', '')
        os.makedirs(os.path.join(save_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'pos_items'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'neg_items'), exist_ok=True)

        pairs_vqa_correctness_rules_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_pairs_vqa_correctness_rules.jsonl')
        pairs_vqa_format_rules_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_pairs_vqa_format_rules.jsonl')

        if not args.overwrite and os.path.exists(pairs_vqa_correctness_rules_save_path):
            print(f'skip {filename}')
            continue

        data_path = os.path.join(args.data_dir, filename)
        with open(data_path) as file:
            lines = file.readlines()

        print(f'preprocess {filename}, {len(lines)=}, {args.max_lines=}')
        lines = lines[:args.max_lines]
        mode = get_mode(filename)

        global NUM_PAIRS_PER_KEY
        if len(lines) > 500000:
            NUM_PAIRS_PER_KEY = min(args.num_pairs_per_key, 3)
        else:
            NUM_PAIRS_PER_KEY = args.num_pairs_per_key

        pos_id2item, neg_id2item, neg_format_id2item, all_incorrect_keys = build_neg_based_on_correctness(lines, mode=mode)

        gt_data_path = None
        for key in gt_path_map:
            if key in filename:
                gt_data_path = gt_path_map[key]
                break

        if gt_data_path is not None:
            print(f'[{filename}] Include gt data path: {gt_data_path}')
            with open(gt_data_path) as file:
                gt_lines = file.readlines()
            gt_pos_id2item, _, _, _ = build_neg_based_on_correctness(gt_lines, mode=mode)

            for key in gt_pos_id2item:
                if key in pos_id2item and not args.force:
                    continue
                pos_id2item[key].extend(gt_pos_id2item[key].copy())

        save_items(
            pos_id2item,
            os.path.join(save_dir, 'pos_items', f'{ds_name}.jsonl'),
        )
        save_items(
            neg_id2item,
            os.path.join(save_dir, 'neg_items', f'{ds_name}.jsonl'),
            question_only=True,
            all_incorrect_keys=all_incorrect_keys,
        )

        save_pairs(
            build_pairs_based_on_pos_neg(pos_id2item=pos_id2item, neg_id2item=neg_id2item, allow_entailment=args.use_correctness_cache),
            pairs_vqa_correctness_rules_save_path,
        )
        save_pairs(
            build_pairs_based_on_pos_neg(pos_id2item=pos_id2item, neg_id2item=neg_format_id2item, allow_entailment=args.use_correctness_cache),
            pairs_vqa_format_rules_save_path,
        )
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='')
    parser.add_argument('--max-lines', type=int, default=int(1e6))
    parser.add_argument('--num-pairs-per-key', type=int, default=15)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--answer-fix', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--use-correctness-cache', action='store_true', default=False)
    args = parser.parse_args()

    NUM_PAIRS_PER_KEY = args.num_pairs_per_key
    main(args)

    print(f'Finish, {NUM_PAIRS_PER_KEY=}')
