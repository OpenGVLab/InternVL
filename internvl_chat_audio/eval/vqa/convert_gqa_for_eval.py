import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str)
parser.add_argument('--dst', type=str)
args = parser.parse_args()

all_answers = []
data = json.load(open(args.src))
for res in data:
    question_id = res['questionId']
    answer = res['answer'].rstrip('.').lower()
    all_answers.append({'questionId': question_id, 'prediction': answer})

with open(args.dst, 'w') as f:
    json.dump(all_answers, f)
