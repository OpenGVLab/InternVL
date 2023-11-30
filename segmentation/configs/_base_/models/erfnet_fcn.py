# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ERFNet',
        in_channels=3,
        enc_downsample_channels=(16, 64, 128),
        enc_stage_non_bottlenecks=(5, 8),
        enc_non_bottleneck_dilations=(2, 4, 8, 16),
        enc_non_bottleneck_channels=(64, 128),
        dec_upsample_channels=(64, 16),
        dec_stages_non_bottleneck=(2, 2),
        dec_non_bottleneck_channels=(64, 16),
        dropout_ratio=0.1,
        init_cfg=None),
    decode_head=dict(
        type='FCNHead',
        in_channels=16,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
