from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from .llama_rmsnorm_monkey_patch import \
    replace_llama_rmsnorm_with_fused_rmsnorm

__all__ = ['replace_llama_attn_with_flash_attn',
           'replace_llama_rmsnorm_with_fused_rmsnorm']
