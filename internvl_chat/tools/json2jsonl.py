
import argparse
import json

argparse = argparse.ArgumentParser()
argparse.add_argument('path', type=str)

args = argparse.parse_args()

assert args.path.endswith('.json')

data = json.load(open(args.path))
writer = open(args.path.replace('.json', '.jsonl'), 'w')
for item in data:
    conversations = item['conversations']
    if conversations[0]['from'] == 'system':
        item['conversations'] = item['conversations'][1:]
    writer.write(json.dumps(item, ensure_ascii=False) + '\n')

writer.close()
