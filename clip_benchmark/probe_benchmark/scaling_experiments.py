import os

from clip_benchmark.cli import get_parser_args, run

if __name__ == '__main__':

    models = ['ViT-B-32-quickgelu,laion400m_e32',
              'ViT-B-32,openai',
              'ViT-B-32,laion2b_s34b_b79k',
              'ViT-B-16,laion400m_e32',
              'ViT-B-16-plus-240,laion400m_e32',
              'ViT-B-16,openai',
              'ViT-L-14-336,openai',
              'ViT-L-14,openai',
              'ViT-B-32,laion2b_e16',
              'ViT-L-14,laion400m_e32',
              'ViT-L-14,laion2b_s32b_b82k',
              'ViT-H-14,laion2b_s32b_b79k',
              'ViT-g-14,laion2b_s12b_b42k',
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

    if not os.path.exists('probe_benchmark/data'):
        os.mkdir('probe_benchmark/data')

    for dataset in datasets:
        dataset_root = 'datasets/' + dataset.split('/')[-1]  # TODO: change!
        print(dataset_root)
        for model_info in models:
            model_info_split = model_info.split(',')
            model, pretrained = model_info_split[0], model_info_split[1]
            for epochs in epoch_vals:
                # For VTAB, do not run >= 25 shot.
                for k in ks:
                    if k >= 25 and dataset.startswith('vtab'):
                        continue
                    for lr in lrs:
                        for bs in batch_sizes:
                            args = get_parser_args()
                            args.dataset_root = dataset_root
                            args.dataset = dataset
                            args.task = 'linear_probe'
                            args.pretrained = pretrained
                            args.model = model
                            args.output = f'probe_benchmark/data/' + f'{model}-{pretrained}-{dataset}-{epochs}-{k}-{lr}-{bs}.json'.replace(
                                '/', '_')
                            if os.path.exists(args.output):
                                print('skipping - exists.')
                                continue
                            args.fewshot_k = k
                            args.fewshot_epochs = epochs
                            args.fewshot_lr = lr
                            args.batch_size = bs
                            run(args)
                            print(dataset, model, pretrained, epochs, k, lr, bs)
