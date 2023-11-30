import argparse

import torch

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

model = torch.load(args.filename, map_location=torch.device('cpu'))
model = model['module']
print(model.keys())

# new_dict = {'state_dict': new_state_dict}
torch.save(model, args.filename.replace('.pt', '_release.pt'))
