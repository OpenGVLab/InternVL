import json
import os
import argparse
from copy import deepcopy

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoTokenizer


def compute_l2_distance(model1, model2):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    total_l2 = 0.0
    total_params = 0

    common_keys = set(state_dict1.keys()) & set(state_dict2.keys())

    for key in common_keys:
        t1 = state_dict1[key].float().cpu()
        t2 = state_dict2[key].float().cpu()

        if t1.shape != t2.shape:
            print(f"⚠️ Shape mismatch at key: {key}, skipping.")
            continue

        diff = t1 - t2
        l2 = torch.norm(diff, p=2)
        total_l2 += l2.item()
        total_params += diff.numel()

    print(f"\n✅ Total L2 distance: {total_l2:.6f}")
    print(f"✅ Average per-parameter L2: {total_l2 / total_params:.8f}" if total_params > 0 else '⚠️ No matching parameters.')

    return total_l2


def convert_keys_back(hf_state_dict):
    new_state_dict = {}

    # Temporary buffer for QKV parts, separated into weight and bias
    qkv_buffer = {}

    for key, value in hf_state_dict.items():
        # === 1. multi_modal_projector → mlp1.*
        if key.startswith('multi_modal_projector.layer_norm.'):
            new_key = key.replace('multi_modal_projector.layer_norm.', 'mlp1.0.')
        elif key.startswith('multi_modal_projector.linear_1.'):
            new_key = key.replace('multi_modal_projector.linear_1.', 'mlp1.1.')
        elif key.startswith('multi_modal_projector.linear_2.'):
            new_key = key.replace('multi_modal_projector.linear_2.', 'mlp1.3.')

        # === 2. embeddings ===
        elif key == 'vision_tower.embeddings.cls_token':
            new_key = 'vision_model.embeddings.class_embedding'
        elif key.startswith('vision_tower.embeddings.patch_embeddings.projection.'):
            new_key = key.replace(
                'vision_tower.embeddings.patch_embeddings.projection',
                'vision_model.embeddings.patch_embedding'
            )
        elif key == 'vision_tower.embeddings.position_embeddings':
            new_key = 'vision_model.embeddings.position_embedding'

        # === 3. encoder.layer.X → encoder.layers.X
        elif key.startswith('vision_tower.encoder.layer.'):
            parts = key.split('.')
            layer_id = parts[3]
            suffix = '.'.join(parts[4:])
            base = f"vision_model.encoder.layers.{layer_id}."

            # Handle QKV weight and bias separately
            if suffix in {
                'attention.q_proj.weight', 'attention.k_proj.weight', 'attention.v_proj.weight',
                'attention.q_proj.bias', 'attention.k_proj.bias', 'attention.v_proj.bias'
            }:
                if layer_id not in qkv_buffer:
                    qkv_buffer[layer_id] = {'weight': {}, 'bias': {}}

                if suffix.endswith('.weight'):
                    if 'q_proj' in suffix:
                        qkv_buffer[layer_id]['weight']['q_proj'] = value
                    elif 'k_proj' in suffix:
                        qkv_buffer[layer_id]['weight']['k_proj'] = value
                    elif 'v_proj' in suffix:
                        qkv_buffer[layer_id]['weight']['v_proj'] = value

                elif suffix.endswith('.bias'):
                    if 'q_proj' in suffix:
                        qkv_buffer[layer_id]['bias']['q_proj'] = value
                    elif 'k_proj' in suffix:
                        qkv_buffer[layer_id]['bias']['k_proj'] = value
                    elif 'v_proj' in suffix:
                        qkv_buffer[layer_id]['bias']['v_proj'] = value

                continue  # Postpone concatenation

            elif suffix.startswith('attention.projection_layer.'):
                new_key = base + 'attn.proj.' + suffix.split('.')[-1]
            elif suffix.startswith('layernorm_before.'):
                new_key = base + 'norm1.' + suffix.split('.')[-1]
            elif suffix.startswith('layernorm_after.'):
                new_key = base + 'norm2.' + suffix.split('.')[-1]
            elif suffix == 'lambda_1':
                new_key = base + 'ls1'
            elif suffix == 'lambda_2':
                new_key = base + 'ls2'
            else:
                new_key = base + suffix

        else:
            new_key = key

        new_state_dict[new_key] = value

    # === 4. Concatenate QKV weights and biases ===
    for layer_id, qkv_parts in qkv_buffer.items():
        base = f"vision_model.encoder.layers.{layer_id}.attn.qkv"

        # Concatenate weights
        if all(k in qkv_parts['weight'] for k in ('q_proj', 'k_proj', 'v_proj')):
            qkv_weight = torch.cat([
                qkv_parts['weight']['q_proj'],
                qkv_parts['weight']['k_proj'],
                qkv_parts['weight']['v_proj']
            ], dim=0)
            new_state_dict[base + '.weight'] = qkv_weight

        # Concatenate biases
        if all(k in qkv_parts['bias'] for k in ('q_proj', 'k_proj', 'v_proj')):
            qkv_bias = torch.cat([
                qkv_parts['bias']['q_proj'],
                qkv_parts['bias']['k_proj'],
                qkv_parts['bias']['v_proj']
            ], dim=0)
            new_state_dict[base + '.bias'] = qkv_bias

    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert HF model weights to original custom key format and compare.")
    parser.add_argument('--custom_path', type=str, required=True, help='Path to custom model config and tokenizer')
    parser.add_argument('--hf_path', type=str, required=True, help='Path to HF-formatted safetensor weights')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save converted model')
    args = parser.parse_args()

    mllm_custom_path = args.custom_path
    mllm_hf_path = args.hf_path
    mllm_save_path = args.save_path

    # Load custom model configuration
    config = AutoConfig.from_pretrained(mllm_custom_path, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)

    # Load HF safetensor weights
    checkpoint_paths = [os.path.join(mllm_hf_path, f) for f in os.listdir(mllm_hf_path) if f.endswith('.safetensors')]
    print(f"\n🔍 Found checkpoint files: {checkpoint_paths}")

    model_state_dict_hf = {}
    for checkpoint_path in checkpoint_paths:
        with safe_open(checkpoint_path, framework='pt') as f:
            for k in f.keys():
                model_state_dict_hf[k] = f.get_tensor(k)

    # Convert key naming style
    model_state_dict = convert_keys_back(model_state_dict_hf)

    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    print(f"\n❌ Missing keys: {missing_keys}")
    print(f"⚠️ Unexpected keys: {unexpected_keys}")

    # Load original model for comparison
    model_compare = AutoModel.from_pretrained(mllm_custom_path, trust_remote_code=True)
    compute_l2_distance(model, model_compare)

    # Save the converted model
    model.save_pretrained(mllm_save_path)

    tokenizer = AutoTokenizer.from_pretrained(mllm_custom_path, trust_remote_code=True)
    tokenizer.save_pretrained(mllm_save_path)
