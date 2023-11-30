import json
import os

import pandas as pd

if __name__ == '__main__':

    compute_df = pd.read_csv('probe_benchmark/clip_table_2.csv')
    # mdf = pd.read_csv("https://gist.githubusercontent.com/mehdidc/58dee67cecd5431a80ee3a2346c9c165/raw/45288ebccaacc34a97f580f8bf16fb3274927f2c/gistfile1.txt")
    mdf = pd.read_csv('probe_benchmark/openclip_results.csv')
    info = []
    # import pdb; pdb.set_trace()
    models = ['ViT-B-32-quickgelu,laion400m_e32',
              'ViT-B-32,openai',
              'ViT-B-32,laion2b_s34b_b79k',
              'ViT-B-16,laion400m_e32',
              'ViT-B-16-plus-240,laion400m_e32',
              'ViT-B-16,openai',
              # 'ViT-L-14-336,openai',
              'ViT-L-14,openai',
              'ViT-B-32,laion2b_e16',
              'ViT-L-14,laion400m_e32',
              'ViT-L-14,laion2b_s32b_b82k',
              'ViT-H-14,laion2b_s32b_b79k',
              'ViT-g-14,laion2b_s12b_b42k',
              ]
    alt_models = ['B/32 400M',
                  'B/32 CLIP WIT',
                  'B/32 2B',
                  'B/16 400M',
                  'B/16+ 400M',
                  'B/16 CLIP WIT',
                  # 'ViT-L-14-336,openai',
                  'L/14 CLIP WIT',
                  'B/32 2B',
                  'L/14 400M',
                  'L/14 2B',
                  'H/14 2B',
                  'g/14 2B',
                  ]

    datasets = ['imagenet1k-unverified', 'cifar100']
    datasets = datasets + [
        'vtab/caltech101',
        'vtab/cifar10',
        'vtab/cifar100',
        'vtab/clevr_count_all',
        'vtab/clevr_closest_object_distance',
        'vtab/diabetic_retinopathy',
        'vtab/dmlab',
        'vtab/dsprites_label_orientation',
        'vtab/dsprites_label_x_position',
        'vtab/dtd',
        'vtab/eurosat',
        'vtab/kitti_closest_vehicle_distance',
        'vtab/flowers',
        'vtab/pets',
        'vtab/pcam',
        'vtab/resisc45',
        'vtab/smallnorb_label_azimuth',
        'vtab/smallnorb_label_elevation',
        'vtab/svhn',
    ]

    ks = [10, 25, -1]
    lrs = [0.1, 0.01, 0.001]
    epoch_vals = [10, 20, 40]
    batch_sizes = [32 * 8]

    def get_us_dataset(pretrained):
        if '2b' in pretrained:
            return 'LAION-2B'
        elif 'laion' in pretrained:
            return 'LAION-400M'
        else:
            return 'CLIP-WIT'

    for dataset in datasets:
        dataset_root = '/datasets01/imagenet_full_size/061417' if dataset.startswith(
            'imagenet') else '/private/home/mitchellw/git/forks/CLIP_benchmark'
        for ii, model_info in enumerate(models):
            model_info_split = model_info.split(',')
            model, pretrained = model_info_split[0], model_info_split[1]
            for epochs in epoch_vals:
                for k in ks:
                    if k == 25 and 'vtab' in dataset:
                        continue
                    for lr in lrs:
                        for bs in batch_sizes:
                            pth = '/private/home/mitchellw/git/forks/CLIP_benchmark/probe_benchmark/data/' + f'{model}-{pretrained}-{dataset}-{epochs}-{k}-{lr}-{bs}.json'.replace(
                                '/', '_')
                            print(pth)
                            assert os.path.exists(pth)
                            row = {
                                'k': k,
                                'lr': lr,
                                'bs': bs,
                                'epochs': epochs,
                                'model': model.replace('-quickgelu', ''),
                                'pretrained': pretrained,
                                'pretrained_short': 'laion2b' if 'laion2b' in pretrained else pretrained,
                                'pretrained_clean': 'LAION' if 'laion' in pretrained else 'CLIP-WiT',
                                'dataset': dataset,
                                'macts': compute_df[compute_df.model == model.replace('-quickgelu', '')][
                                    'image_macts'].values[0],
                                # 'gmacs_total': mdf[mdf.model_fullname_pretty == alt_models[ii]]['gmacs_total'].values[0],
                                # 'samples_seen': mdf[mdf.model_fullname_pretty == alt_models[ii]]['samples_seen'].values[0],
                                'gmacs_total':
                                    mdf[mdf.model_fullname == models[ii].replace(',', ' ')]['gmacs_total'].values[0],
                                'samples_seen':
                                    mdf[mdf.model_fullname == models[ii].replace(',', ' ')]['samples_seen'].values[0],
                                'samples_seen_pretty': mdf[mdf.model_fullname == models[ii].replace(',', ' ')][
                                    'samples_seen_pretty'].values[0],
                                'model_short': models[ii].replace(',', ' '),
                                'upstream_dataset': get_us_dataset(pretrained)
                            }
                            with open(pth, 'r') as f:
                                row.update(json.load(f)['metrics'])
                        info.append(row)

    with open('probe_benchmark/scaling_experiment_data2.json', 'w') as f:
        json.dump(info, f)
