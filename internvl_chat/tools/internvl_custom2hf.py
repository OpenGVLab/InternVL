import json
import os
import argparse
from collections import OrderedDict
from copy import deepcopy

import torch
from safetensors import safe_open
from transformers import (AutoConfig, AutoModel, AutoModelForImageTextToText,
                          AutoTokenizer)


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


def convert_keys_to_hf(custom_state_dict):
    new_state_dict = OrderedDict()
    qkv_split_buffer = {}

    for key, value in custom_state_dict.items():
        # === 1. mlp1.* → multi_modal_projector
        if key.startswith('mlp1.0.'):
            new_key = 'model.' + key.replace('mlp1.0.', 'multi_modal_projector.layer_norm.')
        elif key.startswith('mlp1.1.'):
            new_key = 'model.' + key.replace('mlp1.1.', 'multi_modal_projector.linear_1.')
        elif key.startswith('mlp1.3.'):
            new_key = 'model.' + key.replace('mlp1.3.', 'multi_modal_projector.linear_2.')

        # === 2. embeddings ===
        elif key == 'vision_model.embeddings.class_embedding':
            new_key = 'model.vision_tower.embeddings.cls_token'
        elif key.startswith('vision_model.embeddings.patch_embedding'):
            new_key = 'model.' + key.replace(
                'vision_model.embeddings.patch_embedding',
                'vision_tower.embeddings.patch_embeddings.projection'
            )
        elif key == 'vision_model.embeddings.position_embedding':
            new_key = 'model.vision_tower.embeddings.position_embeddings'

        # === 3. encoder ===
        elif key.startswith('vision_model.encoder.layers.'):
            parts = key.split('.')
            layer_id = parts[3]
            suffix = '.'.join(parts[4:])
            base = f"model.vision_tower.encoder.layer.{layer_id}."

            if suffix.startswith('attn.qkv.weight'):
                qkv_split_buffer[(layer_id, 'weight')] = value
                continue
            elif suffix.startswith('attn.qkv.bias'):
                qkv_split_buffer[(layer_id, 'bias')] = value
                continue
            elif suffix.startswith('attn.proj.'):
                new_key = base + 'attention.projection_layer.' + suffix.split('.')[-1]
            elif suffix.startswith('norm1.'):
                new_key = base + 'layernorm_before.' + suffix.split('.')[-1]
            elif suffix.startswith('norm2.'):
                new_key = base + 'layernorm_after.' + suffix.split('.')[-1]
            elif suffix == 'ls1':
                new_key = base + 'lambda_1'
            elif suffix == 'ls2':
                new_key = base + 'lambda_2'
            else:
                new_key = base + suffix

        # === 4. language_model.model. → language_model.
        elif key == 'language_model.lm_head.weight' or key == 'language_model.model.lm_head.weight':
            new_key = 'lm_head.weight'

        elif key.startswith('language_model.model.'):
            new_key = 'model.' + key.replace('language_model.model.', 'language_model.')

        # === 5. already has model. prefix or default
        elif key.startswith('model.'):
            new_key = key

        else:
            new_key = 'model.' + key

        new_state_dict[new_key] = value

    # === 6. Split QKV ===
    for (layer_id, typ), tensor in qkv_split_buffer.items():
        d = tensor.shape[0] // 3
        q, k, v = tensor[:d], tensor[d:2 * d], tensor[2 * d:]
        base = f"model.vision_tower.encoder.layer.{layer_id}.attention."
        if typ == 'weight':
            new_state_dict[base + 'q_proj.weight'] = q
            new_state_dict[base + 'k_proj.weight'] = k
            new_state_dict[base + 'v_proj.weight'] = v
        else:
            new_state_dict[base + 'q_proj.bias'] = q
            new_state_dict[base + 'k_proj.bias'] = k
            new_state_dict[base + 'v_proj.bias'] = v

    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert custom safetensors weights and compare with HuggingFace model.")
    parser.add_argument('--custom_path', type=str, required=True, help='Path to original safetensors checkpoint folder')
    parser.add_argument('--hf_path', type=str, required=True, help='Path to pretrained HuggingFace model')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the converted model')
    args = parser.parse_args()

    mllm_custom_path = args.custom_path
    mllm_hf_path = args.hf_path
    mllm_save_path = args.save_path

    # Load custom model configuration
    config = AutoConfig.from_pretrained(mllm_hf_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True).to('cuda')

    # Load HF safetensor weights
    checkpoint_paths = [os.path.join(mllm_custom_path, f) for f in os.listdir(mllm_custom_path) if f.endswith('.safetensors')]
    print(f"\n🔍 Found checkpoint files: {checkpoint_paths}")

    model_state_dict_hf = {}
    for checkpoint_path in checkpoint_paths:
        with safe_open(checkpoint_path, framework='pt') as f:
            for k in f.keys():
                model_state_dict_hf[k] = f.get_tensor(k)

    # Convert key naming style
    model_state_dict = convert_keys_to_hf(model_state_dict_hf)

    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    print(f"\n❌ Missing keys: {missing_keys}")
    print(f"⚠️ Unexpected keys: {unexpected_keys}")

    # Load original model for comparison
    model_compare = AutoModelForImageTextToText.from_pretrained(mllm_hf_path, trust_remote_code=True)
    compute_l2_distance(model, model_compare)

    # Save the converted model
    model.save_pretrained(mllm_save_path)

    tokenizer = AutoTokenizer.from_pretrained(mllm_hf_path, trust_remote_code=True)
    tokenizer.save_pretrained(mllm_save_path)
