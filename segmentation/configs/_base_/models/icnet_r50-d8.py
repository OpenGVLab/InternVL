# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ICNet',
        backbone_cfg=dict(
            type='ResNetV1c',
            in_channels=3,
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),
        in_channels=3,
        layer_channels=(512, 2048),
        light_branch_middle_channels=32,
        psp_out_channels=512,
        out_channels=(64, 256, 256),
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    neck=dict(
        type='ICNeck',
        in_channels=(64, 256, 256),
        out_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        channels=128,
        num_convs=1,
        in_index=2,
        dropout_ratio=0,
        num_classes=19,
        norm_cfg=norm_cfg,
        concat_input=False,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=128,
            num_convs=1,
            num_classes=19,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=128,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
