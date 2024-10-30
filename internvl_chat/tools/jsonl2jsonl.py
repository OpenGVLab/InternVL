import argparse
import json
import os

argparse = argparse.ArgumentParser()
argparse.add_argument('path', type=str)

args = argparse.parse_args()

assert args.path.endswith('.jsonl')

f = open(args.path)
data = [json.loads(line) for line in f.readlines()]
writer = open(args.path.replace('.jsonl', '_new.jsonl'), 'w')
for idx, item in enumerate(data):
    item['id'] = idx
    conversations = item['conversations']
    if conversations[0]['from'] == 'system':
        item['conversations'] = item['conversations'][1:]
    writer.write(json.dumps(item, ensure_ascii=False) + '\n')

writer.close()
