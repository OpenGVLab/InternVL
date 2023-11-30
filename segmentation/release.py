import argparse

import torch

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

model = torch.load(args.filename, map_location=torch.device('cpu'))
model = model['module']

# new_model = {}
# for k, v in model.items():
#     if "backbone.blocks" in k:
#         continue
#     if "auxiliary_head" in k:
#         continue
#     if "pos_embed" in k or "patch_embed" in k or "cls_token" in k:
#         continue
#     try:
#         if "bn" in k:
#             print("fp32:", k)
#             new_model[k] = v
#         else:
#             new_model[k] = v
#     except:
#         new_model[k] = v
# print(new_model.keys())

# new_dict = {'state_dict': new_state_dict}
torch.save(model, args.filename.replace('.pt', '_release.pt'))
