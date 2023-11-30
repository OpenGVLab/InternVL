import argparse
import json

try:
    from .tools import VQAEval
except:
    from tools import VQAEval


def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--file-path', type=str, default='results/updated_datasets_231221114523.json')
    args = parser.parse_args()
    return args


def main(args):
    data = json.loads(open(args.file_path).read())
    overall_score = 0
    results = {}
    dataset_names = ['Visual_Reasoning', 'Visual_Perception', 'Visual_Knowledge_Acquisition',
                     'Visual_Commonsense', 'Object_Hallucination']
    for item in data:
        task_type = item['image_path'].split('/')[-2]
        assert task_type in dataset_names
        if task_type in results:
            results[task_type].append(item)
        else:
            results[task_type] = [item]

    for k, v in results.items():
        eval = VQAEval()
        correct = 0
        num = 0
        for i in range(len(v)):
            gt_answers = v[i]['gt_answers']
            answer = v[i]['answer']
            if eval.evaluate(answer, gt_answers) == 1:
                correct += 1
            num += 1
        overall_score += float(correct) / num
        print(f'{k}: {float(correct) / num}')
    print(f'Overall: {overall_score}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
