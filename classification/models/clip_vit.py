# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from transformers import CLIPModel


def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


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


class CLIPViT(nn.Module):

    def __init__(self, patch_size=14, img_size=336, pretrain_size=336, embed_dim=1024, num_heads=16,
                 mlp_ratio=4, depth=48, with_cp=True, freeze_vit=True, cls_target='cls_patch_concat',
                 num_classes=1000, pretrained=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pretrain_size = pretrain_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.cls_target = cls_target
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        model = CLIPModel.from_pretrained(pretrained)
        model.post_layernorm = nn.Identity()
        self.model = model.vision_model

        if freeze_vit:
            _freeze_params(self)

        if cls_target == 'cls_patch_concat':
            self.norm = nn.SyncBatchNorm(embed_dim * 2, eps=1e-6)
            self.head = nn.Linear(embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()
        elif cls_target == 'attention_pooling':
            self.attn_pooling = AttentionPoolingBlock(
                dim=embed_dim, num_heads=num_heads, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=embed_dim)
            self.norm = nn.SyncBatchNorm(embed_dim, eps=1e-6)
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            raise NotImplementedError

        if type(self.head) != nn.Identity:
            self.head.weight.data.normal_(mean=0.0, std=0.01)
            self.head.bias.data.zero_()

    @property
    def dtype(self):
        return self.model.embeddings.patch_embedding.weight.dtype

    def forward_features(self, x):
        x = x.type(self.dtype)
        x = self.model(x)
        x = x.last_hidden_state
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.cls_target == 'cls_patch_concat':
            x = torch.cat((x[:, 0, :], x[:, 1:, :].mean(dim=1)), dim=-1)
        elif self.cls_target == 'attention_pooling':
            x = self.attn_pooling(x)
        else:
            raise NotImplementedError
        x = self.norm(x)
        x = self.head(x)
        return x

    @torch.jit.ignore
    def lr_decay_keywords(self, decay_ratio=0.95):
        lr_ratios = {}

        # layers
        for idx in range(self.depth):
            tag = 'layers.{}.'.format(idx)
            decay = 1.0 * (decay_ratio ** (self.depth - idx))
            lr_ratios[tag] = decay

        # patch_embedding
        lr_ratios['patch_embedding'] = 1.0 * (decay_ratio ** (self.depth + 1))
        lr_ratios['position_embedding'] = 1.0 * (decay_ratio ** (self.depth + 1))
        lr_ratios['pre_layrnorm'] = 1.0 * (decay_ratio ** (self.depth + 1))

        return lr_ratios
