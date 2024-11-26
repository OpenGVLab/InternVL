import os
import json
import argparse

ref_meta = {}
ref_meta_path_list = [
    '/mnt/petrelfs/wangweiyun/workspace_wwy/InternVL-RL-DPO/internvl_chat_dev/data_sampled_lmdeploy_prefix/max_tiles_6_with_inst/internvl_sft_internvl2_8b_dynamic_res_sft_cotv0_clean/meta.json',
    '/mnt/petrelfs/wangweiyun/workspace_wwy/InternVL-RL-DPO/internvl_chat_dev/dpo_data_lmdeploy/internvl_sft_internvl2_8b_dynamic_res_sft_cotv0_v2_clean/meta.json',
    '/mnt/petrelfs/wangweiyun/workspace_wwy/InternVL-RL-DPO/internvl_chat_dev/shell/data/data_finetune_v129.json',
    '/mnt/petrelfs/wangweiyun/workspace_wwy/InternVL-RL-DPO/internvl_chat_dev/shell/data/temp.json',
]

for ref_meta_path in ref_meta_path_list:
    with open(ref_meta_path) as file:
        ref_meta.update(json.load(file))

IMG_DIR_LIST = [
    'langchao:s3://mm_dataset/ocr_data/TextVQA/train_images/',
    'langchao:s3://mm_dataset/ocr_data/ST-VQA/',
    'langchao:s3://ocr/ocr_data/InfoVQA/infographicVQA_train_v1.0_images/',
    'langchao:s3://multi_modal/playground/data/chartqa/',
    'langchao:s3://internvl2/datasets/SROIE/',
    'langchao:s3://multi_modal/ScienceQA/',
    'langchao:s3://multi_modal/CLEVR/',
    'langchao:s3://multi_modal/FigureQA/',
    'langchao:s3://multi_modal/Geometry3K/',
    'langchao:s3://multi_modal/playground/data/docvqa/',
    'langchao:s3://multi_modal/GEOS/',
    'langchao:s3://internvl2/datasets/ai2diagram/ai2d/',
    'langchao:s3://multi_modal/MapQA/',
    'langchao:s3://multi_modal/coco/train2014/',
    'langchao:s3://mm_dataset/gqa/images/',
    'langchao:s3://internvl2/datasets/wildvision/images/',
    'langchao:s3://SA-1B/',
    'langchao:s3://multi_modal/playground/data/dvqa/',
    'langchao:s3://xc_mirror/OpenDataLab___KVQA/',
    'langchao:s3://multi_modal/playground/data/geoqa+/',
    'langchao:s3://multi_modal/Geometry3K/',
    'langchao:s3://multi_modal/UniGeo/',
    'langchao:s3://multi_modal/GEOS/',
    'langchao:s3://internvl2/datasets/geomverse/images/',
    'langchao:s3://internvl2/datasets/mavis/',
    'langchao:s3://internvl2/datasets/reflux/serve_images/',
    'langchao:s3://multi_modal/serve_images/',
    'langchao:s3://multi_modal/DataReflow/raw_data/internvl_chat_llava_0212_To_0510/serve_images/',
    'langchao:s3://multi_modal/DataReflow/raw_data/internvl_chat_llava_0510_To_0523/serve_images/',
    'langchao:s3://multi_modal/DataReflow/raw_data/',
    '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/iconqa/',
    '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/mmmu_data_0701/images/',
    '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/M3CoT/',
    '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/EduChat-Math/Images/Train_Images/',
    '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/GeomVerse/',
    '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/Geo170K/images/',
    '/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets_vision_only/',
    'langchao:s3://internvl2/datasets/mmmu/',
    'langchao:s3://internvl2/datasets/',
    'langchao:s3://internvl2/',
]

for i in range(len(IMG_DIR_LIST)):
    if not IMG_DIR_LIST[i].endswith('/'):
        IMG_DIR_LIST[i] = IMG_DIR_LIST[i] + '/'

def find_prefix(line):
    item = json.loads(line)
    res = ''
    for prefix in IMG_DIR_LIST:
        item['image'] = item['image'].replace('wenhaitmp:s3://internvl/', 'langchao:s3://internvl2/')
        if item['image'].startswith(prefix) and prefix.startswith(res):
            res = prefix
    return res

def clean_prefix(lines, prefix):
    new_lines = []
    for line in lines:
        item = json.loads(line)
        item['image'] = item['image'].replace('wenhaitmp:s3://internvl/', 'langchao:s3://internvl2/')
        assert item['image'].startswith(prefix)
        item['image'] = item['image'][len(prefix):]

        if item.get('is_tie', False):
            continue

        new_lines.append(json.dumps(item) + '\n')

    return new_lines

def save_new_lines(lines, save_path):
    with open(save_path, 'w') as file:
        for line in lines:
            file.write(line)

def main(args):
    meta = {}
    for filename in sorted(os.listdir(args.data_dir)):
        if not filename.endswith('.jsonl'):
            continue

        if filename.endswith('_gpt.jsonl'):
            continue

        if not args.force and os.path.exists(os.path.join(args.save_dir, filename)):
            print(f'skip {filename}')
            continue

        print(f'process {filename}')

        with open(os.path.join(args.data_dir, filename)) as file:
            lines = file.readlines()

        if len(lines) == 0:
            print(f'skip {filename}, {len(lines)=}')
            continue

        prefix = find_prefix(lines[0])
        new_lines = clean_prefix(lines, prefix)
        save_new_lines(new_lines, os.path.join(args.save_dir, filename))

        if not prefix:
            print(f'[Warning] Fail to find prefix: {filename}')

        assert filename.endswith('.jsonl')
        ds_name = filename[:-len('.jsonl')]

        if filename.replace('.jsonl', '') in ref_meta:
            ref_prefix = ref_meta[filename.replace('.jsonl', '')]['root']
        elif filename.replace('_filtered.jsonl', '') in ref_meta:
            ref_prefix = ref_meta[filename.replace('_filtered.jsonl', '')]['root']
        elif filename.replace('_extracted.jsonl', '') in ref_meta:
            ref_prefix = ref_meta[filename.replace('_extracted.jsonl', '')]['root']
        elif not prefix:
            print(f'[Warning] Fail to find ref_prefix: {filename}')
            ref_prefix = None

        meta[f'{ds_name}{args.suffix}'.strip()] = {
            'root': prefix if prefix else ref_prefix,
            'annotation': os.path.join(args.save_dir, filename),
            'data_augment': False,
            'repeat_time': 1,
            'length': len(new_lines),
        }

    save_path = os.path.join(args.save_dir, 'meta.json')
    with open(save_path, 'w') as file:
        json.dump(meta, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()
    if args.data_dir.endswith('/'):
        args.data_dir = args.data_dir[:-1]
    args.save_dir = os.path.join(args.data_dir, 'clean')
    args.data_dir = os.path.join(args.data_dir, 'raw')

    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
