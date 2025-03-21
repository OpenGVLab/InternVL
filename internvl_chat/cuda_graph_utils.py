import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from internvl.model.internvl_chat import InternVLChatModel
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2DecoderLayer,
    Qwen2RotaryEmbedding,
    Qwen2Config,
    repeat_kv,
    apply_rotary_pos_emb,
    flash_attn_func,
    flash_attn_varlen_func,
    pad_input,
    _get_unpad_data,
    index_first_axis,
    unpad_input
)
import argparse
from types import MethodType
class Config:
    def __init__(self):
        self.hidden_size = 896
        self.num_attention_heads = 14
        self.num_key_value_heads = 2
        self.max_position_embeddings = 32768
        self.rope_theta = 1000000.0
        self.attention_dropout = 0.0
        self.rope_scaling = None

class Attn(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self._flash_attn_uses_top_left_mask = False

    def clone_weights_from_module(self, module):
        self.q_proj.weight.data = module.q_proj.weight.data.detach()
        self.q_proj.bias.data = module.q_proj.bias.data.detach()
        self.k_proj.weight.data = module.k_proj.weight.data.detach()
        self.k_proj.bias.data = module.k_proj.bias.data.detach()
        self.v_proj.weight.data = module.v_proj.weight.data.detach()
        self.v_proj.bias.data = module.v_proj.bias.data.detach()
        self.o_proj.weight.data = module.o_proj.weight.data.detach()
        # self.rotary_emb = module.rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # if "padding_mask" in kwargs:
        #     warnings.warn(
        #         "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        #     )

        #     # overwrite attention_mask with padding_mask
        #     attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        # input_dtype = query_states.dtype
        # if input_dtype == torch.float32:
        #     if torch.is_autocast_enabled():
        #         target_dtype = torch.get_autocast_gpu_dtype()
        #     # Handle the case where the model is quantized
        #     elif hasattr(self.config, "_pre_quantization_dtype"):
        #         target_dtype = self.config._pre_quantization_dtype
        #     else:
        #         target_dtype = self.q_proj.weight.dtype

        #     query_states = query_states.to(target_dtype)
        #     key_states = key_states.to(target_dtype)
        #     value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            None,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=False,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Decide whether to use SWA or not by layer index.
        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

def pack_model4cuda_graph(model: InternVLChatModel, batch_size: int, seq_len: int):
    def packed_decoder_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            # attention_mask=attention_mask,
            position_ids,
            # past_key_value=past_key_value,
            # output_attentions=output_attentions,
            # use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs
    
    for decoder_layer in model.language_model.model.layers:
        layer: Qwen2DecoderLayer = decoder_layer
        origin_attn = layer.self_attn
        self_attn = Attn(config=origin_attn.config)
        self_attn = self_attn.to(torch.bfloat16).to(torch.cuda.current_device())
        self_attn.clone_weights_from_module(origin_attn)
        
        hidden_size = origin_attn.config.hidden_size
        dummy_hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=torch.cuda.current_device(), dtype=torch.bfloat16, requires_grad=True)
        dummy_position_ids = torch.tensor([list(range(seq_len)) for _ in range(batch_size)], device=torch.cuda.current_device(), dtype=torch.long)
        self_attn = torch.cuda.make_graphed_callables(self_attn, (dummy_hidden_states, dummy_position_ids))
        decoder_layer.self_attn = self_attn
        decoder_layer.forward = MethodType(packed_decoder_layer_forward, decoder_layer)

if __name__ == '__main__':

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--graph', action='store_true', default=False)
    args = parser.parse_args()
    config = Config()
    a=Attn(config)
    a.apply(init_weights).to(torch.bfloat16).to(0)
    b, s, h = 1, 1024, 896
    hidden_states = torch.randn(b, s, h, device=0, dtype=torch.bfloat16)
    position_ids = torch.tensor([list(range(s)) for _ in range(b)], device=0, dtype=torch.long)
    if args.graph:
        print("use cuda_graph")
        a=torch.cuda.make_graphed_callables(a, (hidden_states, position_ids))
    else:
        print("baseline")
    
    steps=10
    real_hidden_states = [torch.randn(b, s, h, device=0, dtype=torch.bfloat16) for _ in range(steps)]
    real_position_ids = [torch.tensor([list(range(s)) for _ in range(b)], device=0, dtype=torch.long) for _ in range(steps)]
    warmup_iters = 5
    i = 0
    for h, p in zip(real_hidden_states, real_position_ids):
        if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()
        if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(i))
        y=a(h, p)
        torch.cuda.synchronize()
        print(y)
        if i >= warmup_iters: torch.cuda.nvtx.range_pop()
        i+=1
    torch.cuda.cudart().cudaProfilerStop()
# Sample inputs used for capture
# requires_grad state of sample inputs must match
# requires_grad state of real inputs each callable will see.
#x = torch.randn(N, D_in, device='cuda')
#h = torch.randn(N, H, device='cuda', requires_grad=True)
#
#module1 = torch.cuda.make_graphed_callables(module1, (x,))
#module2 = torch.cuda.make_graphed_callables(module2, (h,))
#module3 = torch.cuda.make_graphed_callables(module3, (h,))
#
#real_inputs = [torch.rand_like(x) for _ in range(10)]
#real_targets = [torch.randn(N, D_out, device="cuda") for _ in range(10)]
#
#for data, target in zip(real_inputs, real_targets):
#    optimizer.zero_grad(set_to_none=True)
#
#    tmp = module1(data)  # forward ops run as a graph
#
#    if tmp.sum().item() > 0:
#        tmp = module2(tmp)  # forward ops run as a graph
#    else:
#        tmp = module3(tmp)  # forward ops run as a graph
#
#    loss = loss_fn(tmp, target)
#    # module2's or module3's (whichever was chosen) backward ops,
#    # as well as module1's backward ops, run as graphs
#    loss.backward()
#    optimizer.step()
