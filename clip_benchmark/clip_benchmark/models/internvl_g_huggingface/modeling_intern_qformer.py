from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from peft import LoraConfig, get_peft_model
from timm.models.layers import DropPath
from torch import nn
from transformers import GenerationConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_intern_qformer import (InternQformerConfig,
                                           InternVisionConfig)
from .modeling_qllama import LlamaForCausalLM, _expand_mask, _make_causal_mask

try:
    from .flash_attention import FlashAttention  # v1/v2
except:
    print('FlashAttention is not installed.')

logger = logging.get_logger(__name__)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        out = x.mul_(self.gamma) if self.inplace else x * self.gamma
        return out


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm')
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning('discovered apex but it failed to load, falling back to InternRMSNorm')
    pass


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        return embeddings


class InternVideoEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_frames = getattr(self.config, 'num_frames', 8)
        self.frame_stride = getattr(self.config, 'frame_stride', 2)

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv3d(
            in_channels=3, out_channels=self.embed_dim,
            kernel_size=(self.frame_stride, self.patch_size, self.patch_size),
            stride=(self.frame_stride, self.patch_size, self.patch_size)
        )

        self.num_patches = int(self.num_frames // self.frame_stride) * (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        return embeddings


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()
        mixed_qkv = self.qkv(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, embed_dim)
        query_states, key_states, value_states = mixed_qkv.unbind(2)

        if self.qk_normalization:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = self._shape(query_states, tgt_len, bsz)  # bsz, self.num_heads, seq_len, self.head_dim
        key_states = self._shape(key_states, tgt_len, bsz)  # bsz, self.num_heads, seq_len, self.head_dim
        value_states = self._shape(value_states, tgt_len, bsz)  # bsz, self.num_heads, seq_len, self.head_dim
        mixed_qkv = torch.stack([query_states, key_states, value_states],
                                dim=2)  # bsz, self.num_heads, 3, seq_len, self.head_dim
        context_layer, _ = self.inner_attn(mixed_qkv)
        context_layer = context_layer.flatten(2)
        outputs = self.proj(context_layer)
        return outputs


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states)) * self.ls1)

        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)

        return hidden_states


@dataclass
class InternQformerModelOutput(ModelOutput):
    """
    Class defining the outputs of [`InternQformerModelOutput`].
    """

    loss: Optional[torch.FloatTensor] = None
    loss_itm: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_itg: Optional[torch.FloatTensor] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ['loss', 'loss_itm', 'loss_itc', 'loss_itg']
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class InternPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = InternQformerConfig
    base_model_prefix = 'intern'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r'position_ids',
    ]
    _no_split_modules = ['InternAttention', 'LlamaDecoderLayer', 'LlamaForCausalLM']
    _skip_keys_device_placement = 'past_key_values'
    _keep_in_fp32_modules = ['wo']

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, InternVisionEmbeddings):
            if hasattr(self.config, 'vision_config'):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InternVisionEncoder):
            module.gradient_checkpointing = value


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()

        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class InternVisionModel(InternPreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = InternVisionConfig

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.video_embeddings = InternVideoEmbeddings(config)

        self.encoder = InternVisionEncoder(config)

        # self.post_init()

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        print('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def get_video_embeddings(self):
        return self.video_embeddings

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            elif len(pixel_values.shape) == 5:
                hidden_states = self.video_embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return torch.stack(output, 0)

    @staticmethod
    def backward(ctx, grads):
        input, = ctx.saved_tensors
        dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class InternQformerModel(InternPreTrainedModel):
    config_class = InternQformerConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: InternQformerConfig):
        super().__init__(config)

        text_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size
        clip_embed_dim = config.clip_embed_dim
        attn_pool_num_heads = config.attn_pool_num_heads
        config.text_config.num_query_token = config.num_query_token
        self.num_query_token = config.num_query_token
        self.max_txt_len = config.max_txt_len
        self.label_smoothing = config.label_smoothing

        self.vision_model = InternVisionModel(config.vision_config)  # frozen
        self.qformer = LlamaForCausalLM(config.text_config)  # frozen
        self.query_tokens = nn.Parameter(  # trainable
            torch.zeros(1, config.num_query_token, text_hidden_size)
        )

        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, clip_embed_dim))  # frozen
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # trainable
        self.clip_projector = AttentionPoolingBlock(  # frozen
            dim=vision_hidden_size, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)
        self.clip_projector2 = AttentionPoolingBlock(  # trainable
            dim=text_hidden_size, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)
        self.itm_head = nn.Linear(text_hidden_size, 2)  # trainable
        self.gradient_checkpointing = True

        # Initialize weights and apply final processing
        # self.post_init()

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora)
        if config.use_qformer_lora:
            self.wrap_qformer_lora(r=config.use_qformer_lora)
        if config.force_image_size:
            self.vision_model.resize_pos_embeddings(
                old_size=config.vision_config.image_size,
                new_size=config.force_image_size,
                patch_size=config.vision_config.patch_size
            )

    def wrap_backbone_lora(self, r=32, lora_alpha=16, lora_dropout=0.1):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_qformer_lora(self, r=32, lora_alpha=16, lora_dropout=0.1):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.qformer = get_peft_model(self.qformer, lora_config)
        self.qformer.print_trainable_parameters()

    def get_input_embeddings(self):
        return self.qformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.qformer.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.qformer.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.qformer.get_output_embeddings()

    def get_encoder(self):
        return self.qformer.get_encoder()

    def get_decoder(self):
        return self.qformer.get_decoder()

    @torch.no_grad()
    def _prepare_blip_attention_mask(
            self,
            image_attention_mask: torch.LongTensor,
            attention_mask: torch.LongTensor,
            input_embeds: torch.FloatTensor,
            repeat_time: int,
    ):
        # itm, itc, itg
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        expand_mask = _expand_mask(attention_mask, input_embeds.dtype).to(
            input_embeds.device)  # [bsz, 1, tgt_seq_len, src_seq_len]
        itm_mask, itc_mask, itg_mask = torch.chunk(expand_mask, repeat_time, dim=0)

        itc_mask[:, :, :self.num_query_token, self.num_query_token:] = torch.finfo(input_embeds.dtype).min
        itc_mask[:, :, self.num_query_token:, :self.num_query_token] = torch.finfo(input_embeds.dtype).min
        itc_mask_causal = _make_causal_mask(
            (itc_mask.shape[0], itc_mask.shape[2] - self.num_query_token),
            input_embeds.dtype,
            device=input_embeds.device
        )
        # use causal mask for text in itc
        itc_mask[:, :, self.num_query_token:, self.num_query_token:] += itc_mask_causal

        itg_mask_causal = _make_causal_mask(
            (itg_mask.shape[0], itg_mask.shape[2]),
            input_embeds.dtype,
            device=input_embeds.device
        )
        itg_mask = itg_mask + itg_mask_causal
        itg_mask[:, :, :self.num_query_token, :self.num_query_token] = 0
        attention_mask = torch.cat([itm_mask, itc_mask, itg_mask], dim=0)

        return attention_mask

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            positive_input_ids: torch.FloatTensor,
            positive_attention_mask: torch.LongTensor,
            negative_input_ids: torch.FloatTensor,
            negative_attention_mask: torch.LongTensor,
            summarize_input_ids: torch.FloatTensor,
            summarize_attention_mask: torch.LongTensor,
            input_ids: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            labels: torch.LongTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InternQformerModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        image_embeds = vision_outputs[0]
        backbone_embeds = self.clip_projector(image_embeds)

        # step 2: prepare input_ids and attention_mask for three sub-tasks:
        # 1) image-text matching; 2) image-text contrastive learning; 3) image-grounded text generation.
        batch_size = input_ids.shape[0]
        self.positive_num = batch_size // 2
        input_ids = torch.cat([negative_input_ids[:-self.positive_num], positive_input_ids[-self.positive_num:],
                               summarize_input_ids, input_ids], dim=0)  # [3 * batch_size, seq_len]
        itm_attention_mask = torch.cat(
            [negative_attention_mask[:-self.positive_num], positive_attention_mask[-self.positive_num:]], dim=0)
        selected = itm_attention_mask.sum(1) - 1
        attention_mask = torch.cat(
            [itm_attention_mask, summarize_attention_mask, attention_mask], dim=0)  # [3 * batch_size, seq_len]

        repeat_time = input_ids.size(0) // batch_size
        # step 3: forward the input_ids and attention_mask through the text encoder.
        input_embeds = self.get_input_embeddings()(input_ids)
        query_tokens = self.query_tokens.repeat(repeat_time * batch_size, 1, 1)
        input_embeds = torch.cat([query_tokens, input_embeds], dim=1)
        image_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        attention_mask = self._prepare_blip_attention_mask(
            image_attention_mask, attention_mask, input_embeds, repeat_time
        )
        if type(self.qformer.model) == LlamaForCausalLM:
            outputs = self.qformer.model.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=image_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                repeat_time=repeat_time,
            ).last_hidden_state
        else:
            outputs = self.qformer.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=image_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                repeat_time=repeat_time,
            ).last_hidden_state
        image_embeds = outputs[:, :self.num_query_token]
        text_embeds = outputs[:, self.num_query_token:]
        image_itm, image_itc, image_itg = image_embeds.chunk(repeat_time, dim=0)
        text_itm, text_itc, text_itg = text_embeds.chunk(repeat_time, dim=0)

        ###============== Image-Text Matching ===================###
        image_itm = self.itm_head(image_itm)
        logits = image_itm.mean(dim=1)
        itm_labels = torch.cat([
            torch.zeros(batch_size - self.positive_num, dtype=torch.long, device=logits.device),
            torch.ones(self.positive_num, dtype=torch.long, device=logits.device)
        ], dim=0)
        itm_labels[selected == 1] = -100  # ignore empty texts
        loss_itm = F.cross_entropy(logits, itm_labels)
        neg_match_acc = ((logits[:batch_size - self.positive_num].argmax(dim=-1) == 0) / (
                batch_size - self.positive_num)).sum()
        pos_match_acc = ((logits[-self.positive_num:].argmax(dim=-1) == 1) / self.positive_num).sum()

        ###============== Image-Text Contrastive ===================###
        image_itc = self.clip_projector2(image_itc)

        selected = summarize_attention_mask.sum(1) - 1
        text_itc = text_itc[torch.arange(text_itc.shape[0]), selected]
        text_itc = text_itc @ self.text_projection

        # normalized features
        image_itc = image_itc / image_itc.norm(dim=1, keepdim=True)
        text_itc = text_itc / text_itc.norm(dim=1, keepdim=True)
        image_itc_all = GatherLayer.apply(image_itc).flatten(0, 1)
        text_itc_all = GatherLayer.apply(text_itc).flatten(0, 1)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        sim_i2t = logit_scale * (image_itc @ text_itc_all.t())
        sim_t2i = logit_scale * (text_itc @ image_itc_all.t())
        bs = image_itc.size(0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long, device=sim_i2t.device)
        targets[selected == 4] = -100  # ignore empty texts
        loss_itc = (
                           F.cross_entropy(sim_i2t, targets, label_smoothing=self.label_smoothing)
                           + F.cross_entropy(sim_t2i, targets, label_smoothing=self.label_smoothing)
                   ) / 2

        ###============== Image-grounded Text Generation ===================###
        logits = self.qformer.lm_head(text_itg)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.qformer.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_itg = F.cross_entropy(shift_logits, shift_labels)

        vision_sim = F.cosine_similarity(backbone_embeds.detach(), image_itc).mean()

        loss = loss_itm + loss_itc + loss_itg
        if dist.get_rank() == 0:
            print(f'loss: {loss.item()}, loss_itm: {loss_itm.item()}, loss_itc: {loss_itc.item()}, '
                  f'loss_itg: {loss_itg.item()}, vision_similarity: {round(vision_sim.item(), 5)}, '
                  f'logit scale: {round(1.0 / logit_scale.item(), 5)}, '
                  f'pos_match_acc: {round(pos_match_acc.item(), 4)}, '
                  f'neg_match_acc: {round(neg_match_acc.item(), 4)}')
        return InternQformerModelOutput(
            loss=loss,
            loss_itc=loss_itc.detach(),
            loss_itm=loss_itm.detach(),
            loss_itg=loss_itg.detach(),
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        image_embeds = vision_outputs[0]

        batch_size = image_embeds.shape[0]
        input_embeds = self.get_input_embeddings()(input_ids)
        query_tokens = self.query_tokens.repeat(batch_size, 1, 1)
        input_embeds = torch.cat([query_tokens, input_embeds], dim=1)
        image_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        outputs = self.qformer.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            vision_hidden_states=image_embeds,
            generation_config=generation_config,
            use_zero_attention_mask=True,
            **generate_kwargs,
        )

        return outputs

    def get_text_features(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            text_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that
                contains the language model logits, the past key values and the hidden states if
                `output_hidden_states=True`.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.get_input_embeddings()(input_ids)
        attention_mask = _expand_mask(attention_mask, input_embeds.dtype).to(
            input_embeds.device)  # [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask += _make_causal_mask(
            (attention_mask.shape[0], attention_mask.shape[2]),
            input_embeds.dtype,
            device=input_embeds.device
        )
        if type(self.qformer.model) == LlamaForCausalLM:
            outputs = self.qformer.model.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=None,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        else:
            outputs = self.qformer.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=None,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        return outputs

    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        image_embeds = vision_outputs[0]
        backbone_embeds = image_embeds

        batch_size = image_embeds.shape[0]
        input_embeds = self.query_tokens.repeat(batch_size, 1, 1)

        attention_mask = torch.ones(input_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        attention_mask = _expand_mask(attention_mask, input_embeds.dtype).to(
            input_embeds.device)  # [bsz, 1, tgt_seq_len, src_seq_len]
        if type(self.qformer.model) == LlamaForCausalLM:
            outputs = self.qformer.model.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=image_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        else:
            outputs = self.qformer.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=image_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        return backbone_embeds, outputs


class InternQformerModelForBenchmark(InternQformerModel):
    """
    This class is used for model benchmarking with the clip-benchmark codebase.
    """

    def encode_image_text(self, image_embeds, input_ids):
        attention_mask = input_ids > 0
        batch_size = image_embeds.shape[0]

        input_embeds = self.get_input_embeddings()(input_ids)
        query_tokens = self.query_tokens.repeat(batch_size, 1, 1)
        input_embeds = torch.cat([query_tokens, input_embeds], dim=1)
        image_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        attention_mask = _expand_mask(attention_mask, input_embeds.dtype).to(
            input_embeds.device)  # [bsz, 1, tgt_seq_len, src_seq_len]
        outputs = self.qformer.model.forward_train(
            inputs_embeds=input_embeds,
            vision_hidden_states=image_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        image_embeds = outputs[:, :self.num_query_token]
        # image_itm = self.itm_head(image_embeds)
        # logits = image_itm.mean(dim=1).to(torch.float32).softmax(dim=-1)
        # return logits[:, 1].to(image_itm.dtype)
        return image_embeds

    def encode_image(self, image):
        backbone_embeds, image_embeds = self.get_image_features(
            pixel_values=image,
            output_hidden_states=False,
            return_dict=True,
        )
        backbone_embeds = self.clip_projector(backbone_embeds)
        image_embeds = self.clip_projector2(image_embeds)
        # ensemble
        backbone_embeds = backbone_embeds / backbone_embeds.norm(dim=1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        image_embeds = image_embeds + backbone_embeds
        return image_embeds

    def encode_text(self, text):
        attention_mask = text > 0
        text_embeds = self.get_text_features(
            input_ids=text,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        text_embeds = text_embeds[torch.arange(text_embeds.shape[0]), attention_mask.sum(1) - 1]
        text_embeds = text_embeds @ self.text_projection
        return text_embeds

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
