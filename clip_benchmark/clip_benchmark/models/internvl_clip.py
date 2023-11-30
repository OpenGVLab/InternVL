import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from timm.models.layers import DropPath
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

try:
    from timm.models.layers.helpers import to_2tuple
except:
    from timm.layers.helpers import to_2tuple

from functools import partial

import torch.utils.checkpoint as checkpoint
import torchvision.transforms as T
from einops import rearrange
from torchvision.transforms import InterpolationMode
from transformers import LlamaTokenizer

try:
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.modules.mlp import FusedMLP
    from flash_attn.ops.rms_norm import DropoutAddRMSNorm
except:
    print('flash attention is not installed.')


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


class RMSNorm(nn.Module):
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


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, use_fused_rmsnorm=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):

        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, x):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
            fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
            use_fused_rmsnorm=False):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              use_flash_attn=use_flash_attn, causal=False, norm_layer=norm_layer,
                              qk_normalization=qk_normalization,
                              use_fused_rmsnorm=use_fused_rmsnorm)
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def forward(self, x, residual=None):

        def _inner_forward(x, residual=None):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.attn(x)))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.mlp(x)))
                return x, residual
            else:
                assert residual is None
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x

        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class InternVL_CLIP(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.0,
            embed_dim: int = 3200,
            num_heads: int = 25,
            mlp_ratio: int = 4,
            init_values: float = 0.1,
            qk_normalization: bool = True,
            depth: int = 48,
            use_flash_attn: bool = True,
            use_fused_rmsnorm: bool = True,
            use_fused_mlp: bool = True,
            fused_mlp_heuristic: int = 1,
            with_cp: bool = True,
            context_length: int = 80,
            transformer_width: int = 4096,
            transformer_type: str = 'alpaca-7b-chinese',
            llama_path: str = None,
            use_lora: bool = True,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 768,
            layerscale_no_force_fp32: bool = True):
        super().__init__()

        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, print(
            'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent')

        self.use_flash_attn = use_flash_attn
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.transformer_type = transformer_type
        self.transformer_width = transformer_width

        """ text encoder of InternVL """
        llama_config = LlamaConfig.from_pretrained(llama_path)
        llama_config.causal = True
        llama_config.use_flash_attention = use_flash_attn
        model = LlamaForCausalLM(llama_config).to(torch.float16)
        if not use_lora:
            self.transformer = model.model
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)
            self.transformer = model.base_model.model.model

        self.transformer.gradient_checkpointing = True
        self.text_projection = nn.Parameter(torch.empty(transformer_width, clip_embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        """ image encoder of InternVL """
        if use_fused_rmsnorm:
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn, use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=with_cp,
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  use_fused_rmsnorm=use_fused_rmsnorm)
            for i in range(depth)])
        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def forward_vit_blocks(self, blocks, x):
        residual = None
        for idx, blk in enumerate(blocks):
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual)
        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual
        return x

    def encode_image(self, image):
        x = self.patch_embed(image.type(self.dtype))
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.forward_vit_blocks(self.blocks, x)
        x = self.clip_projector(x)
        return x

    def encode_text(self, text):
        text_key_padding_mask = text > 0

        x = self.transformer(input_ids=text, attention_mask=text_key_padding_mask).last_hidden_state
        x = x[torch.arange(x.shape[0]), text_key_padding_mask.sum(1) - 1]
        x = x @ self.text_projection
        return x

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

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


class InternVLTokenizer(nn.Module):
    def __init__(self, model_path):
        super(InternVLTokenizer, self).__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = ' '  # allow padding
        self.tokenizer.add_eos_token = True

    def forward(self, text, prefix='summarize:'):
        if type(text) == str:
            text = prefix + text
        elif type(text) == list:
            text = [prefix + item for item in text]
        text = self.tokenizer(text, return_tensors='pt', max_length=80, truncation=True, padding=True).input_ids
        return text


def process_checkpoint(ckpt):
    new_ckpt = {}
    for k, v in ckpt['module'].items():
        if 'bamboo' in k or 'predictor' in k or 'decoder' in k or 'loss' in k:
            continue
        new_k = k.replace('clip.transformer.', 'transformer.')
        new_k = new_k.replace('clip.text_projection', 'text_projection')
        new_k = new_k.replace('clip.logit_scale', 'logit_scale')

        new_ckpt[new_k] = v
    return new_ckpt


def build_transform(task, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if task == 'retrieval':
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])
    else:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)])
    return transform


def get_model_and_transform(task, image_size, device):
    llama_path = '/mnt/petrelfs/share_data/chenzhe1/data/llm/chinese_alpaca_lora_7b'
    model = InternVL_CLIP(img_size=image_size,
                          layerscale_no_force_fp32=False,
                          llama_path=llama_path).to(device)
    transform = build_transform(task, image_size)
    return model, transform


def load_husky_vit6b_checkpoint():
    new_ckpt = {}
    for i in range(4):
        ckpt_path = f'/mnt/petrelfs/share_data/wangwenhai/llm/new_husky_pretrain2/pytorch_model-0000{str(i + 1)}-of-00004.bin'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        for k, v in ckpt.items():
            if 'vision_model.' in k:
                new_k = k.replace('vision_model.embeddings.class_embedding', 'cls_token')
                new_k = new_k.replace('vision_model.embeddings.position_embedding', 'pos_embed')
                new_k = new_k.replace('vision_model.embeddings.patch_embedding.weight', 'patch_embed.proj.weight')
                new_k = new_k.replace('vision_model.embeddings.patch_embedding.bias', 'patch_embed.proj.bias')
                new_k = new_k.replace('vision_model.encoder.layers', 'blocks')
                new_k = new_k.replace('.ls1', '.ls1.gamma')
                new_k = new_k.replace('.ls2', '.ls2.gamma')
                new_k = new_k.replace('vision_model.', '')
                new_k = new_k.replace('vision_model.', '')
                new_ckpt[new_k] = v
    return new_ckpt


def load_internvl_clip(model_name, pretrained, cache_dir, device, task):
    llm_path = '/mnt/petrelfs/share_data/chenzhe1/data/llm/chinese_alpaca_lora_7b'
    tokenizer = InternVLTokenizer(llm_path)
    model, transform = get_model_and_transform(task=task, image_size=224, device=device)
    ckpt_path = f'/mnt/petrelfs/wangwenhai/workspace_swj/code/sim_clip_wj_deepspeed/exp/configs/' \
                f'6b_vit/6b_vit_{model_name}_clip_alpaca_7b_laion5b_peak_1e-5_256gpu_all_trainable_degradation.sh/' \
                f'{pretrained}/mp_rank_00_model_states.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    new_ckpt = process_checkpoint(ckpt)
    message = model.load_state_dict(new_ckpt, strict=False)
    return model, transform, tokenizer


if __name__ == '__main__':
    llama_path = '/mnt/petrelfs/share_data/chenzhe1/data/llm/chinese_alpaca_lora_7b'

    tokenizer = LlamaTokenizer.from_pretrained(llama_path)
    tokenizer.pad_token = ' '  # allow padding
    tokenizer.add_eos_token = True

    image = torch.rand(1, 3, 224, 224).cuda()

    text = ['a diagram', 'a dog', 'a cat']
    text = tokenizer(text, return_tensors='pt').input_ids.cuda()
    model = InternVL_CLIP(llama_path=llama_path).cuda()

    # image_feats = model.encode_image(image)
    # text_feats = model.encode_text(text)

    # 这里有两个模型权重，请根据需要选择：
    # 第一个是Laion-5B直出的模型，ImageNet zero-shot acc=82.7
    ckpt_path = '/mnt/petrelfs/share_data/wangwenhai/internvl/' \
                '6b_vit_exp101_clip_alpaca_7b_laion5b_peak_1e-5_256gpu_all_trainable_degradation.sh/' \
                '699/mp_rank_00_model_states.pt'
    # 第二个是经过小规模高质量数据精调的模型，ImageNet zero-shot acc=83.2
    ckpt_path = '/mnt/petrelfs/share_data/wangwenhai/internvl/' \
                '6b_vit_exp126_clip_alpaca_7b_laion5b_peak_1e-5_256gpu_all_trainable_degradation.sh/' \
                '1499/mp_rank_00_model_states.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    new_ckpt = process_checkpoint(ckpt)
    message = model.load_state_dict(new_ckpt, strict=False)
    print(message)
    # merge lora
    model.transformer.merge_and_unload()
    state_dict = model.state_dict()
    torch.save('intern_clip_merge_lora.pth', state_dict)
