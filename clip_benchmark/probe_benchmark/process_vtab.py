import json

import pandas as pd

# make a new version of vtab

if __name__ == '__main__':
    df = pd.read_json('probe_benchmark/scaling_experiment_data2.json')
    df = df[df.fewshot_k == -1]
    datasets = [
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
    all_info = []
    for n, g in df.groupby(['model', 'pretrained', 'samples_seen_pretty']):
        count = 0
        total = 0.
        for d in datasets:
            g_filter = g[g.dataset == d]
            count += 1
            total += g_filter.lp_acc1.max()

        avg = total / count
        info = {'dataset': 'vtab', 'lp_acc1': avg, 'fewshot_k': -1}
        for k in ['model', 'pretrained', 'upstream_dataset', 'gmacs_total', 'samples_seen_pretty']:
            info[k] = g[k].values[0]
        all_info.append(info)

    with open('probe_benchmark/scaling_experiment_data_vtab.json', 'w') as f:
        json.dump(all_info, f)
