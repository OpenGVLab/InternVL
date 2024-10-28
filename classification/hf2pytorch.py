import argparse
import os

import torch
from safetensors.torch import load_file as safetensors_load_file

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process and convert model state_dicts.')
parser.add_argument('input_dir', type=str, help='Directory containing input .bin and .safetensors files.')
parser.add_argument('output_file', type=str, help='Output file to save the converted state_dict.')
args = parser.parse_args()

# Verify that the input directory exists
if not os.path.isdir(args.input_dir):
    raise ValueError(f'Input directory does not exist: {args.input_dir}')

# List all files in the input directory
filenames = os.listdir(args.input_dir)

# Filter files to include only .bin and .safetensors files
filenames = [f for f in filenames if f.endswith('.bin') or f.endswith('.safetensors')]
filepaths = [os.path.join(args.input_dir, f) for f in filenames]
print(f'Found files: {filenames}')

# Initialize an empty state_dict to store the loaded data
state_dict = {}

# Loop over each file and load its contents
for filepath in filepaths:
    print(f'Loading: {filepath}')
    if filepath.endswith('.bin'):
        # Load .bin file using torch.load
        loaded_dict = torch.load(filepath, map_location='cpu')
        state_dict.update(loaded_dict)
    elif filepath.endswith('.safetensors'):
        # Load .safetensors file using safetensors
        loaded_dict = safetensors_load_file(filepath, device='cpu')
        state_dict.update(loaded_dict)
    else:
        raise ValueError(f'Unsupported file format: {filepath}')

# Print the keys of the loaded state_dict
print(f'Loaded state_dict keys: {list(state_dict.keys())}')

# Initialize a new state_dict to store the converted data
new_state_dict = {}

# Iterate over each key-value pair in the original state_dict
for k, v in state_dict.items():
    # Replace key substrings according to specified mapping rules
    k_new = k
    k_new = k_new.replace('embeddings.class_embedding', 'cls_token')
    k_new = k_new.replace('embeddings.position_embedding', 'pos_embed')
    k_new = k_new.replace('embeddings.patch_embedding.weight', 'patch_embed.proj.weight')
    k_new = k_new.replace('embeddings.patch_embedding.bias', 'patch_embed.proj.bias')
    k_new = k_new.replace('ls1', 'ls1.gamma')
    k_new = k_new.replace('ls2', 'ls2.gamma')
    k_new = k_new.replace('encoder.layers.', 'blocks.')
    # Update the new_state_dict with the transformed key and original value
    new_state_dict[k_new] = v

# Print the keys of the converted state_dict
print(f'Converted state_dict keys: {list(new_state_dict.keys())}')

# Wrap the new_state_dict in a dictionary under the 'module' key
new_dict = {'module': new_state_dict}

# Save the new_dict to the specified output file
print(f'Saving converted state_dict to: {args.output_file}')
torch.save(new_dict, args.output_file)
