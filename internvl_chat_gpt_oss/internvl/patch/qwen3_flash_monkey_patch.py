# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch

from types import MethodType
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask

_use_top_left_mask = flash_attn_supports_top_left_mask()

def _forward_qwen3(
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

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        raise NotImplementedError(f"Inference mode is not implemented, please switch to eager attention.")
        # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    query_length = query_states.shape[2]

    # B,N_h,L,D_h --> B,L,N_h,D_h
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask.squeeze(0)

    with torch.no_grad():
        max_seqlen = max([
            cu_seqlens[idx+1] - cu_seqlens[idx]
            for idx in range(cu_seqlens.size(0) - 1)
        ]).item()

    causal = self.is_causal and not (_use_top_left_mask and query_length == 1)
    use_sw = self.sliding_window and key_states.shape[1] > self.sliding_window
    flash_kwargs = {"window_size": (self.sliding_window, self.sliding_window)} if use_sw else {}
    flash_kwargs["dropout_p"] = self.attention_dropout

    if "softcap" in kwargs:
        flash_kwargs["softcap"] = kwargs.get("softcap")

    if "s_aux" in kwargs:
        flash_kwargs["s_aux"] = kwargs.get("s_aux")

    attn_output = flash_attn_varlen_func(
        q=query_states,
        k=key_states,
        v=value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=self.scaling,
        causal=causal,
        **flash_kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def replace_qwen3_attention_class(model):
    for layer in model.model.layers:
        print('Flash sink attn (w. varlen) applied to Qwen3/Qwen3-MoE')
        layer.self_attn.forward = MethodType(_forward_qwen3, layer.self_attn)
