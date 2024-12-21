import argparse
import json
import os
import random
import re
from collections import defaultdict

from eval.vqa.textvqa_eval import TextVQAAccuracyEvaluator

random.seed(0)

evaluator_cache = {}
evaluator = TextVQAAccuracyEvaluator()
option_candidate = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

# you can download this file from this url: https://huggingface.co/datasets/Weiyun1025/M3CoT-ScienceQA-Format/blob/main/train_pair_with_res.jsonl
gt_path_map = {
    'm3cot': 'M3CoT/train_pair_with_res.jsonl',
}


def merge_dict(*dict_list):
    merged_dict = defaultdict(list)
    for curr_dict in dict_list:
        for key in curr_dict:
            merged_dict[key].extend(curr_dict[key])
    return merged_dict


def parse_answer(response):
    answer_trigger = 'Final answer:'
    if response.count(answer_trigger) == 0:
        answer_trigger = 'Final Answer:'
    if response.count(answer_trigger) == 0:
        answer_trigger = '答案:'

    assert response.count(answer_trigger) <= 2, f'Fail to find Answer, {response.count(answer_trigger)=}'
    assert response.count('\n') >= 2, f'Fail to find rationale, {response=}'

    rationale, answer = response.rsplit(answer_trigger, 1)
    assert len(rationale.strip()) > 0, f'Empty rationale:\n{response}'
    assert '\n' not in answer.strip(), f'Answer with multiple paragraphs:\n{answer}'

    return rationale.strip(), answer.strip()


def isfloat(x):
    try:
        float(x)
        return True
    except:
        return False


def math_score(target: str, prediction: str, max_relative_change: float = 1e-3) -> bool:
    def _to_float(text: str) -> float:
        text = text.replace('degrees', '')
        text = text.replace('degree', '')
        text = text.replace('\\angle', '')
        text = text.replace('degrees', '')
        text = text.replace('°', '')
        text = text.replace('%', '')
        text = text.replace('cm', '')

        try:
            return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> float:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                # return float(text.rstrip('%')) / 100.0
                return float(text.rstrip('%'))
            else:
                return float(text)
        except ValueError:
            return None

    if len(target) == 4 and target.startswith('20'):
        return prediction.lower() == target.lower()

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def multi_choice_score(answer_pred, answer_gt):
    answer_pred = answer_pred.strip()
    answer_gt = answer_gt.strip()
    if answer_pred.lower() == answer_gt.lower():
        return 1

    if len(answer_pred) >= 2 and answer_pred[1] == '.':
        answer_pred = answer_pred[0]

    if len(answer_pred) >= 3 and answer_pred[0] == '(' and answer_pred[2] == ')':
        answer_pred = answer_pred[1]

    return answer_pred.lower() == answer_gt.lower()


def check_answer(answer_pred, answer_gt, mode):
    if (answer_pred, answer_gt) in evaluator_cache:
        accuracy = evaluator_cache[(answer_pred, answer_gt)]

    if answer_pred.lower() == answer_gt.lower():
        return 1

    accuracy = 0

    # vqa_score
    if 'vqa_score' in mode:
        merged_outputs = [
            {
                'pred_answer': answer_pred,
                'gt_answers': [answer_gt] * 10,
            },
        ]
        accuracy = max(accuracy, evaluator.eval_pred_list(merged_outputs, disable_tqdm=True))

        if len(evaluator.answer_processor(answer_pred)) == 0:
            accuracy = 0

        if len(evaluator.answer_processor(answer_gt)) == 0:
            accuracy = 0

    # relaxed_accuracy (e.g. charqa)
    if 'relaxed_accuracy' in mode:
        accuracy = max(accuracy, relaxed_correctness(answer_gt, answer_pred))

    # anls (e.g. docvqa, infographicsvqa)
    if 'anls' in mode:
        gt_answer = ' '.join(answer_gt.strip().lower().split())
        det_answer = ' '.join(answer_pred.strip().lower().split())
        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer_gt.upper()), len(answer_pred.upper()))
        accuracy = max(accuracy, float(dist) / float(length))

    if 'mc_score' in mode:
        accuracy = max(accuracy, multi_choice_score(answer_pred, answer_gt))

    if 'math_score' in mode:
        accuracy = max(accuracy, math_score(answer_pred, answer_gt))

    accuracy = int(accuracy > 0.9)
    evaluator_cache[(answer_pred, answer_gt)] = accuracy
    return accuracy


def _contain_keywords(ds_name, keywords):
    for keyword in keywords:
        if keyword in ds_name:
            return True
    return False


def _get_mode(ds_name):
    if _contain_keywords(ds_name, ['chartqa']):
        return ['relaxed_accuracy']

    if _contain_keywords(ds_name, ['docvqa', 'infographics']):
        return ['anls']

    if _contain_keywords(ds_name, ['SROIE', 'CLEVR_math', 'geos', 'geometry']):
        return ['relaxed_accuracy', 'vqa_score', 'mc_score']

    return ['vqa_score', 'mc_score', 'math_score']


def _fix_answer(item, answer_pred, answer_gt, mc=False):
    answer_pred_orig = answer_pred
    answer_gt_orig = answer_gt
    answer_pred = answer_pred.lower()
    answer_gt = answer_gt.lower()

    if mc:
        try:
            answer_pred = post_process(answer_pred_orig)
        except:
            return item

        answer_gt = answer_gt.upper()
        assert len(answer_pred) == 1
        assert answer_gt in option_candidate

    if (
        answer_gt in answer_pred
        # 30,594 -> 30594
        or answer_gt.strip('.').replace(',', '') in answer_pred.strip('.').replace(',', '')
    ):
        item['response'] = answer_gt_orig.join(item['response'].rsplit(answer_pred_orig, 1))
        item['response'] = item['response'].strip().strip('**').strip()
        _, answer_pred_after_fix = parse_answer(item['response'])
        item['answer_pred'] = answer_pred_after_fix

    other_lines, last_line = item['response'].rsplit('\n', 1)
    if '**Final' in last_line:
        last_line = last_line.replace('**Final', 'Final')
        item['response'] = f'{other_lines}\n{last_line}'.strip()

    return item


def post_process(pred):
    pred = pred.strip().strip('*').strip().upper()

    if len(pred) == 1:
        return pred

    if len(pred) > 1 and not pred[1].isalpha() and pred[0] in option_candidate:
        return pred[0]

    if len(pred) > 2 and pred[0] == '(' and pred[2] == ')' and pred[1] in option_candidate:
        return pred[1]

    raise RuntimeError(f'Fail to parse pred: {pred}')


def is_consistent(meta):
    result = meta['consistency'].split('.')[0]
    result = meta['consistency'].split()[0]
    result = result.strip().strip('.').strip()

    if result.lower() == 'yes':
        return True

    if result.lower() == 'no':
        return False

    raise RuntimeError(meta)


def _build_items_based_on_correctness(lines, mode):
    pos_id2item = defaultdict(list)
    pos_inconsistent_id2item = defaultdict(list)
    neg_id2item = defaultdict(list)
    neg_format_id2item = defaultdict(list)
    for line in lines:
        item = json.loads(line)
        image = str(item['image'])
        question = item['question']
        answer_gt = item['answer']
        response = item['response']

        if 'meta' in item:
            meta = item['meta']
            try:
                consistent = is_consistent(meta)
            except Exception as e:
                print(e)
                continue
        else:
            consistent = True

        if args.use_correctness_cache:
            correct = int(item['is_correct'])
        else:
            try:
                _, answer_pred = parse_answer(response)
                item['answer_pred'] = answer_pred
            except:
                item['answer_pred'] = 'None'
                neg_format_id2item[(image, question, answer_gt)].append(item)
                continue

            if args.answer_fix:
                if (
                    'mc_score' in mode
                    and "Answer with the option's letter from the given choices directly." in question
                ):
                    mc = True
                else:
                    assert "Answer with the option's letter from the given choices directly." not in question
                    mc = False

                item = _fix_answer(item, answer_pred, answer_gt, mc=mc)
                response = item['response']
                answer_pred = item['answer_pred']

            correct = check_answer(answer_pred, answer_gt, mode=mode)

        assert correct in [0, 1], correct

        if correct == 1 and not consistent:
            pos_inconsistent_id2item[(image, question, answer_gt)].append(item)
            continue

        if correct == 1:
            pos_id2item[(image, question, answer_gt)].append(item)
        else:
            neg_id2item[(image, question, answer_gt)].append(item)

    return pos_id2item, pos_inconsistent_id2item, neg_id2item, neg_format_id2item


def build_neg_based_on_correctness(lines, mode):
    pos_id2item, pos_inconsistent_id2item, neg_id2item, neg_format_id2item = _build_items_based_on_correctness(lines, mode=mode)

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
        f'num_inconsistent_pos_samples={sum(len(v) for v in pos_inconsistent_id2item.values())}, '
        f'{all_correct=}, '
        f'{all_incorrect=}, '
    )
    return pos_id2item, pos_inconsistent_id2item, neg_id2item, neg_format_id2item, all_incorrect_keys


def _build_pair_based_on_pos_neg(item_pos, item_neg):
    image_pos = item_pos['image']
    question_pos = item_pos['question']
    answer_gt_pos = item_pos['answer']
    response_pos = item_pos['response']

    image_neg = item_neg['image']
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
                items_set.add((str(item['image']), item['question'], item['answer']))

        items_list = []
        for item in items_set:
            try:
                image = eval(item[0])
            except:
                image = item[0]
            items_list.append({
                'image': image,
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
        filtered_pair = {
            'image': image,
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
    for filename in os.listdir(args.data_dir):
        if not filename.endswith('.jsonl'):
            continue

        save_dir = args.save_dir
        ds_name = os.path.basename(filename).replace('.jsonl', '')
        os.makedirs(os.path.join(save_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'pos_items'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'neg_items'), exist_ok=True)

        pairs_vqa_correctness_rules_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_pairs_vqa_correctness_rules.jsonl')
        pairs_vqa_correctness_rules_and_claims_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_pairs_vqa_correctness_rules_and_claims.jsonl')
        pairs_vqa_format_rules_save_path = os.path.join(save_dir, 'raw', f'{ds_name}_pairs_vqa_format_rules.jsonl')

        if not args.overwrite and os.path.exists(pairs_vqa_correctness_rules_save_path):
            print(f'skip {filename}')
            continue

        data_path = os.path.join(args.data_dir, filename)
        with open(data_path) as file:
            lines = file.readlines()

        for extra_data_dir in args.extra_data_dir:
            extra_data_path = os.path.join(extra_data_dir, filename)
            if os.path.exists(extra_data_path):
                print(f'[{filename}] Add {extra_data_path}')
                with open(extra_data_path) as file:
                    lines.extend(file.readlines())

        print(f'preprocess {filename}, {len(lines)=}, {args.max_lines=}')
        lines = lines[:args.max_lines]
        mode = _get_mode(filename)

        global NUM_PAIRS_PER_KEY
        if len(lines) > 500000:
            NUM_PAIRS_PER_KEY = min(args.num_pairs_per_key, 3)
        else:
            NUM_PAIRS_PER_KEY = args.num_pairs_per_key

        pos_id2item, pos_inconsistent_id2item, neg_id2item, neg_format_id2item, all_incorrect_keys = build_neg_based_on_correctness(lines, mode=mode)

        gt_data_path = None
        for key in gt_path_map:
            if key in filename:
                gt_data_path = gt_path_map[key]
                break

        if gt_data_path is not None:
            print(f'[{filename}] Include gt data path: {gt_data_path}')
            with open(gt_data_path) as file:
                gt_lines = file.readlines()
            gt_pos_id2item, _, _, _, _ = build_neg_based_on_correctness(gt_lines, mode=mode)

            for key in gt_pos_id2item:
                if key in pos_id2item and not args.force:
                    continue
                pos_id2item[key].extend(gt_pos_id2item[key].copy())

        save_items(
            merge_dict(pos_id2item, pos_inconsistent_id2item),
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
    parser.add_argument('--extra-data-dir', nargs='+', type=str, default='')
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
