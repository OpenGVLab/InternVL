# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from .clip_vit import CLIPViT
from .intern_vit_6b import InternViT6B


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'intern_vit_6b':
        model = InternViT6B(
            num_classes=config.MODEL.NUM_CLASSES,
            patch_size=config.MODEL.INTERN_VIT_6B.PATCH_SIZE,
            img_size=config.DATA.IMG_SIZE,
            pretrain_size=config.MODEL.INTERN_VIT_6B.PRETRAIN_SIZE,
            qkv_bias=config.MODEL.INTERN_VIT_6B.QKV_BIAS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            embed_dim=config.MODEL.INTERN_VIT_6B.EMBED_DIM,
            num_heads=config.MODEL.INTERN_VIT_6B.NUM_HEADS,
            mlp_ratio=config.MODEL.INTERN_VIT_6B.MLP_RATIO,
            init_values=config.MODEL.INTERN_VIT_6B.INIT_VALUES,
            qk_normalization=config.MODEL.INTERN_VIT_6B.QK_NORMALIZATION,
            depth=config.MODEL.INTERN_VIT_6B.DEPTH,
            use_flash_attn=config.MODEL.INTERN_VIT_6B.USE_FLASH_ATTN,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            freeze_vit=config.MODEL.INTERN_VIT_6B.FREEZE_VIT,
            pretrained=config.MODEL.INTERN_VIT_6B.PRETRAINED,
            cls_target=config.MODEL.INTERN_VIT_6B.CLS_TARGET,
            norm_type=config.MODEL.INTERN_VIT_6B.NORM_TYPE,
        )
    elif model_type == 'clip_vit':
        model = CLIPViT(
            patch_size=config.MODEL.CLIP_VIT.PATCH_SIZE,
            img_size=config.DATA.IMG_SIZE,
            pretrain_size=config.MODEL.CLIP_VIT.PRETRAIN_SIZE,
            embed_dim=config.MODEL.CLIP_VIT.EMBED_DIM,
            num_heads=config.MODEL.CLIP_VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.CLIP_VIT.MLP_RATIO,
            depth=config.MODEL.CLIP_VIT.DEPTH,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            freeze_vit=config.MODEL.CLIP_VIT.FREEZE_VIT,
            pretrained=config.MODEL.CLIP_VIT.PRETRAINED,
            cls_target=config.MODEL.CLIP_VIT.CLS_TARGET,
        )
    else:
        raise NotImplementedError(f'Unkown model: {model_type}')

    return model
