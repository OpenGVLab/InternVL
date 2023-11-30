norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/vit-b16_p16_224-80ecf9dd.pth', # noqa
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        out_indices=(2, 5, 8, 11),
        final_norm=False,
        with_cls_token=True,
        output_cls_token=True),
    decode_head=dict(
        type='DPTHead',
        in_channels=(768, 768, 768, 768),
        channels=256,
        embed_dims=768,
        post_process_channels=[96, 192, 384, 768],
        num_classes=150,
        readout_type='project',
        input_transform='multiple_select',
        in_index=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
