import os

from clip_benchmark.cli import get_parser_args, run

# /private/home/mitchellw/miniconda3/envs/cb/bin/python probe_benchmark/laion5b_fewshot_experiments.py
if __name__ == '__main__':

    models = ['ViT-B-32-quickgelu,laion400m_e32',
              'ViT-B-32,openai',
              'ViT-B-32,laion2b_s34b_b79k',
              'ViT-B-16,laion400m_e32',
              # 'ViT-B-16-plus-240,laion400m_e32',
              'ViT-B-16,openai',
              # 'ViT-L-14-336,openai',
              'ViT-L-14,openai',
              # 'ViT-B-32,laion2b_e16',
              'ViT-L-14,laion400m_e32',
              'ViT-L-14,laion2b_s32b_b82k',
              'ViT-H-14,laion2b_s32b_b79k',
              ]

    datasets = ['imagenet1k-unverified']

    ks = [1, 2, 4, 8, 16, 32, 64, 128]
    lrs = [0.1, 0.01, 0.001, 0.0001]
    epoch_vals = [10, 20, 40, 80]
    batch_sizes = [32 * 8]

    for epochs in epoch_vals:
        for dataset in datasets:
            dataset_root = '/datasets01/imagenet_full_size/061417' if dataset.startswith(
                'imagenet') else '/private/home/mitchellw/git/forks/CLIP_benchmark'
            for model_info in models:
                model_info_split = model_info.split(',')
                model, pretrained = model_info_split[0], model_info_split[1]

                for k in ks:
                    for lr in lrs:
                        for bs in batch_sizes:
                            args = get_parser_args()
                            args.dataset_root = dataset_root
                            args.dataset = dataset
                            args.task = 'linear_probe'
                            args.pretrained = pretrained
                            args.model = model
                            args.output = '/private/home/mitchellw/git/forks/CLIP_benchmark/probe_benchmark/data/' + f'{model}-{pretrained}-{dataset}-{epochs}-{k}-{lr}-{bs}.json'.replace(
                                '/', '_')
                            if os.path.exists(args.output):
                                print('skipping - exists.')
                            args.fewshot_k = k
                            args.fewshot_epochs = epochs
                            args.fewshot_lr = lr
                            args.batch_size = bs
                            args.skip_load = True  # NOTE
                            run(args)
