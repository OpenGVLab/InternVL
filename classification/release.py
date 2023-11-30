
import argparse

import torch

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

model = torch.load(args.filename, map_location=torch.device('cpu'))
del model['optimizer']
model['model'] = model['model_ema']
del model['model_ema']
del model['amp']

model_ = model['model']
new_model = {}
for k, v in model_.items():
    if k.startswith('norm.'):
        new_model[k] = v
    elif k.startswith('head.'):
        new_model[k] = v
model['model'] = new_model
print(model['model'].keys())

torch.save(model, args.filename.replace('.pth', '_ema.pth'))
