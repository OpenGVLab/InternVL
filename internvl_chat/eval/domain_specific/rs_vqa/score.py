import argparse
import json


def is_correct_count(response, answer):
    try:
        response = int(response) if response is not None else 0
        answer = int(answer)
    except ValueError:
        return False

    if response == 0 and answer == 0:
        return True
    elif 0 < response <= 100 and 0 < answer <= 100:
        return True
    elif 100 < response <= 1000 and 100 < answer <= 1000:
        return True
    elif response > 1000 and answer > 1000:
        return True
    return False


def is_correct_area(response, answer):
    try:
        response = int(response) if response is not None else 0
        answer = int(answer.rstrip('m2'))
    except ValueError:
        return False
    return is_correct_count(response, answer)


def calculate_scores(data):
    type_counts = {}
    type_correct = {}
    for entry in data:
        question_type = entry['question_type']
        response = entry['response']
        answer = entry['gt_answer']

        if question_type not in type_counts:
            type_counts[question_type] = 0
            type_correct[question_type] = 0
        type_counts[question_type] += 1

        if question_type == 'count':
            if is_correct_count(response, answer):
                type_correct[question_type] += 1
        elif question_type == 'area':
            if is_correct_area(response, answer):
                type_correct[question_type] += 1
        else:
            if response and response.lower() == answer.lower():
                type_correct[question_type] += 1

    type_scores = {}
    for question_type in type_counts:
        score = type_correct[question_type] / type_counts[question_type]
        type_scores[question_type] = round(score, 4)

    total_correct = sum(type_correct.values())
    total_count = sum(type_counts.values())
    total_score = round(total_correct / total_count, 4) if total_count > 0 else 0.0

    total_correct_useful = sum([v for k, v in type_correct.items() if k not in ['count', 'area']])
    total_count_useful = sum([v for k, v in type_counts.items() if k not in ['count', 'area']])
    total_score_useful = round(total_correct_useful / total_count_useful, 4) if total_count_useful > 0 else 0.0
    print(f'{type_scores=}')
    print(f'{total_score_useful=}')
    return type_scores, total_score, total_score_useful, type_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='')
    args = parser.parse_args()

    with open(args.output_file, 'r') as f:
        data = json.load(f)
    if 'outputs' in data:
        data = data['outputs']
    type_scores, total_score, total_score_useful, type_counts = calculate_scores(data)

    results = {
        'type_scores': type_scores,
        'type_counts': type_counts,
        'total_score': total_score,
        'total_score_useful': total_score_useful,
        'outputs': data
    }
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
