# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import time
import argparse

import torch
from tqdm import tqdm

from config import get_config
from models import build_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='internimage_t_1k_224')
    parser.add_argument('--ckpt_dir', type=str,
                        default='/mnt/petrelfs/share_data/huangzhenhang/code/internimage/checkpoint_dir/new/cls')
    parser.add_argument('--onnx', default=False, action='store_true')
    parser.add_argument('--trt', default=False, action='store_true')

    args = parser.parse_args()
    args.cfg = os.path.join('./configs', f'{args.model_name}.yaml')
    args.ckpt = os.path.join(args.ckpt_dir, f'{args.model_name}.pth')
    args.size = int(args.model_name.split('.')[0].split('_')[-1])

    cfg = get_config(args)
    return args, cfg

def get_model(args, cfg):
    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location='cpu')['model']

    model.load_state_dict(ckpt)
    return model

def speed_test(model, input):
    # warmup
    for _ in tqdm(range(100)):
        _ = model(input)

    # speed test
    torch.cuda.synchronize()
    start = time.time()
    for _ in tqdm(range(100)):
        _ = model(input)
    end = time.time()
    th = 100 / (end - start)
    print(f"using time: {end - start}, throughput {th}")

def torch2onnx(args, cfg):
    model = get_model(args, cfg).cuda()

    # speed_test(model)

    onnx_name = f'{args.model_name}.onnx'
    torch.onnx.export(model,
                      torch.rand(1, 3, args.size, args.size).cuda(),
                      onnx_name,
                      input_names=['input'],
                      output_names=['output'])

    return model

def onnx2trt(args):
    from mmdeploy.backend.tensorrt import from_onnx

    onnx_name = f'{args.model_name}.onnx'
    from_onnx(
        onnx_name,
        args.model_name,
        dict(
            input=dict(
                min_shape=[1, 3, args.size, args.size],
                opt_shape=[1, 3, args.size, args.size],
                max_shape=[1, 3, args.size, args.size],
            )
        ),
        max_workspace_size=2**30,
    )

def check(args, cfg):
    from mmdeploy.backend.tensorrt.wrapper import TRTWrapper

    model = get_model(args, cfg).cuda()
    model.eval()
    trt_model = TRTWrapper(f'{args.model_name}.engine',
                           ['output'])

    x = torch.randn(1, 3, args.size, args.size).cuda()

    torch_out = model(x)
    trt_out = trt_model(dict(input=x))['output']

    print('torch out shape:', torch_out.shape)
    print('trt out shape:', trt_out.shape)

    print('max delta:', (torch_out - trt_out).abs().max())
    print('mean delta:', (torch_out - trt_out).abs().mean())

    speed_test(model, x)
    speed_test(trt_model, dict(input=x))

def main():
    args, cfg = get_args()

    if args.onnx or args.trt:
        torch2onnx(args, cfg)
        print('torch -> onnx: succeess')

    if args.trt:
        onnx2trt(args)
        print('onnx -> trt: success')
        check(args, cfg)

if __name__ == '__main__':
    main()
