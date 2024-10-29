# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import transformers


def replace_llama_rmsnorm_with_fused_rmsnorm():
    try:
        from functools import partial

        from apex.normalization import FusedRMSNorm
        LlamaRMSNorm = partial(FusedRMSNorm, eps=1e-6)   # noqa
        transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
        print('Discovered apex.normalization.FusedRMSNorm - will use it instead of LlamaRMSNorm')
    except ImportError:
        # using the normal LlamaRMSNorm
        pass
    except Exception:
        print('discovered apex but it failed to load, falling back to LlamaRMSNorm')
        pass
