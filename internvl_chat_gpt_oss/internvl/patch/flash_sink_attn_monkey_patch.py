# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# This file contains code originally written by Wenhao Li.
# Modified and maintained by Weiyun Wang.
# --------------------------------------------------------

import torch

from types import MethodType
from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

from .flash_sink_attn import flash_sink_attn_func, flash_sink_attn_varlen_func, SlidingCacheManager

def _forward_gpt_oss(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # B,L,N_h,D_h --> B,N_h,L,D_h
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        raise NotImplementedError(f"Inference mode is not implemented, please switch to eager attention.")
        # cache_kwargs = {"cache_position": cache_position}
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    # ================================================
    # NOTE: 最关键的代码
    # 算子仅支持训练，不支持推理，推理请转换为eager attention
    manager = SlidingCacheManager(self.sliding_window)
    manager.update(key_states, value_states)
    attn_output = flash_sink_attn_func(
        query_states,
        key_states,
        value_states,
        self.sinks,
        manager)
    # ================================================

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _forward_gpt_oss_with_varlen(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    if self.cu_seqlens is not None:
        attention_mask = self.cu_seqlens

    # B,L,N_h,D_h --> B,N_h,L,D_h
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        raise NotImplementedError(f"Inference mode is not implemented, please switch to eager attention.")
        # cache_kwargs = {"cache_position": cache_position}
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # B,N_h,L,D_h --> B,L,N_h,D_h
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    # ================================================
    # NOTE: varlen形状转换
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask.squeeze(0)
    # ================================================

    # ================================================
    # NOTE: 最关键的代码
    # 算子仅支持训练，不支持推理，推理请转换为eager attention
    manager = SlidingCacheManager(self.sliding_window)
    manager.update(key_states, value_states)
    attn_output = flash_sink_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        self.sinks,
        cu_seqlens,
        manager)
    # ================================================

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def replace_gpt_oss_with_flash_sink_attn(model, use_varlen=False):
    for layer in model.model.layers:
        if use_varlen:
            print('Flash sink attn (w. varlen) applied to GPT-OSS')
            layer.self_attn.forward = MethodType(_forward_gpt_oss_with_varlen, layer.self_attn)
        else:
            print('Flash sink attn (w/o varlen) applied to GPT-OSS')
            layer.self_attn.forward = MethodType(_forward_gpt_oss, layer.self_attn)
