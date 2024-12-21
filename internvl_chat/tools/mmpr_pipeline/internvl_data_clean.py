import os
import json
from collections import defaultdict

meta_path = '/mnt/petrelfs/wangweiyun/workspace_wwy/open_source/InternVL/internvl_chat/shell/data/dev_mpo/meta_oc_data_241203_with_wh_v9.json'
DATA_DIR = '/mnt/petrelfs/wangweiyun/workspace_wwy/open_source/InternVL/internvl_chat'

def parse_answer(response):
    answer_trigger = 'Final answer:'
    if response.count(answer_trigger) == 0:
        answer_trigger = 'Final Answer:'

    assert response.count(answer_trigger) <= 2, f"Fail to find Answer, {response.count(answer_trigger)=}"
    assert response.count('\n') >= 2, f"Fail to find rationale, {response=}"

    rationale, answer = response.rsplit(answer_trigger, 1)
    assert len(rationale.strip()) > 10, f"Empty rationale:\n{response}"
    assert '\n' not in answer.strip(), f"Answer with multiple paragraphs:\n{answer=}\n{response=}"

    return rationale.strip(), answer.strip()

def check(response):
    try:
        rationale, answer = parse_answer(response)
    except:
        return False
    return len(rationale) > 0 and len(answer) > 0

def main():
    with open(meta_path) as file:
        meta = json.load(file)

    invalid = 0
    for ds_name, ds_info in meta.items():
        if not ds_info['annotation'].startswith('/'):
            ds_info['annotation'] = os.path.join(DATA_DIR, ds_info['annotation'])

        with open(ds_info['annotation']) as file:
            lines = file.readlines()

        info = defaultdict(int)
        filtered_lines = []
        for line in lines:
            item = json.loads(line)
            key = 'chosen' if 'chosen' in item else 'response'

            check_success = check(item[key])
            if (
                ('correctness_rules' in ds_name or 'format_rules' in ds_name)
                and not check_success
                and 'step by step' in item['question']
            ):
                info['not_check_success'] += 1
                continue

            if 'Final answer:..' in item[key]:
                invalid += 1
                info['Final answer:..'] += 1
                continue

            if 'incorrect' in item[key]:
                invalid += 1
                info['incorrect claims'] += 1
                continue

            filtered_lines.append(line)

        if len(info) > 0:
            print(ds_name)
            for k, v in info.items():
                print(k, v)
            print()

        meta[ds_name]['length'] = len(filtered_lines)

        if len(lines) != len(filtered_lines):
            print(f'[{ds_name}] {len(lines)} ==> {len(filtered_lines)}\nsave to {ds_info["annotation"]}')
            print()
            with open(ds_info['annotation'], 'w') as file:
                file.writelines(filtered_lines)

    with open(meta_path, 'w') as file:
        json.dump(meta, file)

    print(f'{invalid=}')

if __name__ == '__main__':
    main()
