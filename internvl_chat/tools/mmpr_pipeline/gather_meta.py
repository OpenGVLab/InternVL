import io
import os
import json
import random
import argparse

from PIL import Image
from petrel_client.client import Client

client = Client()

image_error = set()
image_exist = {}
ceph2local = {
    "langchao:s3://multi_modal/playground/data/chartqa/": "chartqa",
    "langchao:s3://multi_modal/playground/data/geoqa+/": "geoqa_plus",
    "langchao:s3://multi_modal/UniGeo/": "UniGeo",
    "langchao:s3://multi_modal/serve_images/": "DataReflow/unknown",
    "langchao:s3://internvl2/datasets/SROIE/": "SROIE",
    "langchao:s3://multi_modal/FigureQA/": "FigureQA",
    "/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/iconqa/": "iconqa",
    "langchao:s3://ocr/ocr_data/InfoVQA/infographicVQA_train_v1.0_images/": "InfoVQA",
    "langchao:s3://multi_modal/playground/data/dvqa/": "dvqa",
    "langchao:s3://multi_modal/MapQA/": "MapQA",
    "/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/GeomVerse/": "GeomVerse",
    "langchao:s3://internvl2/datasets/ai2diagram/ai2d/": "ai2d",
    "langchao:s3://mm_dataset/gqa/images/": "gqa",
    "/mnt/petrelfs/share_data/wangweiyun/share_data_dpo/jsonl_format/openbmb/RLAIF-V-Dataset/images": "RLAIF-V",
    "langchao:s3://internvl2/datasets/logs-240710-to-240902/serve_images/": "DataReflow/240710_to_240902",
    "langchao:s3://multi_modal/ScienceQA/": "ScienceQA",
    "langchao:s3://SA-1B/": "SA-1B",
    "/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/M3CoT/": "M3CoT",
    "langchao:s3://multi_modal/GEOS/": "GEOS",
    "langchao:s3://multi_modal/DataReflow/raw_data/": "DataReflow",
    "langchao:s3://multi_modal/DataReflow/raw_data/internvl_chat_llava_0212_To_0510/serve_images/": "DataReflow/0212_to_0510",
    "/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/Geo170K/images/": "Geo170K",
    "langchao:s3://multi_modal/CLEVR/": "CLEVR",
    "langchao:s3://internvl2/datasets/wildvision/images/": "wildvision",
    "langchao:s3://multi_modal/coco/train2014/": "coco",
    "langchao:s3://multi_modal/coco/": "coco",
    "langchao:s3://multi_modal/playground/data/vg/": "vg",
    "langchao:s3://multi_modal/DataReflow/raw_data/internvl_chat_llava_0510_To_0523/serve_images/": "DataReflow/0510_to_0523",
    "langchao:s3://multi_modal/playground/data/docvqa/": "docvqa",
    "langchao:s3://multi_modal/Geometry3K/": "Geometry3K",
    "langchao:s3://internvl2/datasets/reflux/serve_images/": "DataReflow",
    "langchao:s3://mm_dataset/ocr_data/TextVQA/train_images/": "TextVQA",
    "langchao:s3://private-dataset-pnorm/study/": "private/study",
    "vc-reader:s3://multi-modal/playground/data/koniq-10k/": "koniq",
    "vc-reader:s3://mm-dataset/gpt4o/": "private/gpt4o",
    "vc-reader:s3://gui/visual_inputs/multi_modal_2024/single_iterate/raw_data/wangbo/schedual_extract/": "gui/schedual_extract",
    "/mnt/petrelfs/wangweiyun/workspace_cz/inhouse_data/mmmu_tiku/": "private/mmmu_tiku",
    "/mnt/petrelfs/share_data/wangweiyun/share_data_sft/datasets/CaraJ/MAVIS-Function/": "MAVIS-Function",
    "langchao:s3://internvl2/datasets/crawler_data2/": "private/crawler_data",
    "langchao:s3://multi_modal/agent_data/AndroidUI/": "gui/AndroidUI",
    "langchao:s3://multi_modal/TabMWP/": "TabMWP",
    "langchao:s3://internvl2/datasets/mmmu/": "mmmu",
    "langchao:s3://internvl2/datasets/": "internvl",
    "vc-reader:s3://internvl2/datasets/": "internvl",
    "langchao:s3://mm_dataset/LLaVAR/images/": "LLaVAR",
}

prefix_to_replace = {
    'MM-Reasoning-Private/images/': '',
    '/mnt/petrelfs/wangweiyun/workspace_wwy/sync/oc_dpo_model_data_241203/data/': 'private/st_241203',
}


def save_items(lines, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as file:
        file.writelines(lines)


def save_image(image_path, save_path):
    if 's3://' in image_path:
        image_path = io.BytesIO(client.get(image_path))
    image = Image.open(image_path).convert('RGB')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


def save_each_image(image, image_dir, save_dir):
    save_path = os.path.join(save_dir, image)
    image_path = os.path.join(image_dir, image)

    if save_path in image_error:
        return

    if save_path not in image_exist:
        image_exist[save_path] = os.path.exists(save_path)

    if not image_exist[save_path]:
        try:
            save_image(image_path, save_path)
            image_exist[save_path] = True
        except KeyboardInterrupt:
            raise
        except:
            image_error.add(image_path)
            print(f'[Rank {rank}] Fail to load {image_path}')


def save_images(lines, image_dir, save_dir, dsname):
    print_freq = max(len(lines) // 100, 1)
    for line_idx, line in enumerate(lines):
        item = json.loads(line)
        image = item['image']

        if isinstance(image, str):
            image = [image]

        for i in image:
            save_each_image(i, image_dir, save_dir)

        if rank == 0 and line_idx % print_freq == 0:
            print(f'[Rank {rank}] [{dsname}] [Progress] {line_idx} / {len(lines)}')


def get_image_save_dir(root):
    if root is None:
        return ''

    if root in ceph2local:
        return ceph2local[root]

    for k, v in prefix_to_replace.items():
        if root.startswith(k):
            return os.path.join(v, root[len(k):].strip('/'))

    raise RuntimeError(root)


def main():
    with open(args.meta_path) as file:
        meta = json.load(file)

    invalid_root = set()
    for info in meta.values():
        root = info['root']
        try:
            get_image_save_dir(root)
        except RuntimeError:
            invalid_root.add(root)

    if invalid_root:
        print('Find invalid roots!')
        for root in invalid_root:
            print(root)
        exit(0)

    meta_new = {}
    num_samples = 0
    num_samples_wo_private = 0
    for ds_idx, (dsname, info) in enumerate(meta.items()):
        root = info['root']
        annotation = info['annotation']
        repeat_time = info['repeat_time']
        length = info['length']

        num_samples += length
        image_save_dir = get_image_save_dir(root)
        image_save_dir = os.path.join(args.save_dir, 'images', image_save_dir)
        items_save_dir = os.path.join(args.save_dir, 'annotations', f'{dsname.replace("/", "_")}.jsonl')

        is_private = (
            'DataReflow' in image_save_dir
            or 'private' in dsname
            or 'private' in image_save_dir
        )

        if root is None:
            image_save_dir = None

        if not args.keep_private_data and is_private:
            continue
        num_samples_wo_private += length

        meta_new[dsname] = info.copy()
        meta_new[dsname]['annotation'] = items_save_dir
        meta_new[dsname]['is_private'] = is_private

        if 'sa1b' not in dsname:
            meta_new[dsname]['root'] = image_save_dir

        with open(annotation) as file:
            lines = file.readlines()

        if repeat_time < 1:
            random.seed(ds_idx)
            lines = random.sample(lines, k=int(len(lines) * repeat_time))
            meta_new[dsname]['repeat_time'] = 1
            meta_new[dsname]['length'] = len(lines)

        if rank == 0:
            print(f'Begin to process {dsname}, {image_save_dir=}, {items_save_dir=}, {len(lines)=}, {len(lines[rank::world_size])=}')
            save_items(lines, items_save_dir)

        if args.save_image and image_save_dir is not None and 'sa1b' not in dsname:
            lines = lines[rank::world_size]
            save_images(lines, root, image_save_dir, dsname=dsname)

    if rank == 0:
        print(f'{num_samples=}, {num_samples_wo_private=}')
        with open(os.path.join(args.save_dir, 'meta.json'), 'w') as file:
            json.dump(meta_new, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-path', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='')
    parser.add_argument('--save-image', action='store_true', default=False)
    parser.add_argument('--keep-private-data', action='store_true', default=False)
    args = parser.parse_args()

    rank = int(os.environ['SLURM_PROCID']) if 'SLURM_PROCID' in os.environ else 0
    world_size = int(os.environ["SLURM_NTASKS"]) if 'SLURM_NTASKS' in os.environ else 1

    if rank == 0:
        print(f'{rank=}, {world_size=}')

    main()

# srun -p Intern5 --gres=gpu:0 --ntasks=256 --ntasks-per-node=8 --cpus-per-task=2 python -u tools/mmpr_pipeline/gather_meta.py --save-image --keep-private-data
# srun -p Intern5 --gres=gpu:0 --ntasks=256 --ntasks-per-node=8 --cpus-per-task=2 python -u tools/mmpr_pipeline/gather_meta.py --save-image --keep-private-data --meta-path /mnt/petrelfs/wangweiyun/workspace_wwy/open_source/InternVL/internvl_chat/shell/data/dev_mpo/meta_oc_data_241203_with_wh_v8.json --save-dir /mnt/petrelfs/wangweiyun/workspace_wwy/open_source/MMPR-Private-241215-v8
