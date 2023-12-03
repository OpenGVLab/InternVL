import pandas as pd

# make a new version of vtab

if __name__ == '__main__':
    df_full = pd.read_json('probe_benchmark/scaling_experiment_data2.json')
    df = df_full[df_full.fewshot_k == -1]
    df25 = df_full[df_full.fewshot_k == 25]
    df10 = df_full[df_full.fewshot_k == 10]

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
        'vtab_svhn',
    ]

    datasets2 = [
        'imagenet1k-unverified', 'cifar100'
    ]

    all_info = []
    cols = []
    first = True
    for n, g in df_full.groupby(['model', 'pretrained', 'samples_seen_pretty']):
        count = 0
        total = 0.
        for d in datasets:
            g_filter = g[(g.dataset == d) & (g.fewshot_k == -1)]
            count += 1
            total += g_filter.lp_acc1.max()

        avg = total / count
        info = {'VTAB acc': avg}
        if first:
            cols.append('VTAB acc')

        for d in datasets2:
            for k in [10, 25, -1]:
                g_filter = g[(g.dataset == d) & (g.fewshot_k == k)]
                info[f'{d}: {k} shot'] = g_filter.lp_acc1.max()
                if first:
                    cols.append(f'{d}: {k} shot')

        for k in ['model', 'pretrained', 'upstream_dataset', 'gmacs_total', 'samples_seen_pretty']:
            info[k] = g[k].values[0]
        all_info.append(info)
        first = False

    df = pd.DataFrame(all_info)
    formatters = {}
    print(df.keys())
    columns = ['model', 'samples_seen_pretty', 'upstream_dataset']
    df = df.sort_values(by=['model', 'samples_seen_pretty', 'upstream_dataset'])
    for ds in cols:
        columns.append(ds)
        formatters[ds] = lambda x: f'{100 * x:.2f}'
    latex = df.to_latex(columns=columns, formatters=formatters)
    print(latex)

    # with open('probe_benchmark/scaling_experiment_data_combined.json', 'w') as f:
    #   json.dump(all_info, f)
