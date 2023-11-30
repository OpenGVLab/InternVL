checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_small_20220308-e638c41c.pth'  # noqa

# model settings
backbone_norm_cfg = dict(type='LN')
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='PCPVT',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=3,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[4, 2, 2, 2],
        strides=[4, 2, 2, 2],
        mlp_ratios=[8, 8, 4, 4],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        norm_after_stage=False,
        drop_rate=0.0,
        attn_drop_rate=0.,
        drop_path_rate=0.2),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
