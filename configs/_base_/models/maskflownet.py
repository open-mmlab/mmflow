MaskflownetS_checkpoint = 'https://download.openmmlab.com/mmflow/maskflownet/maskflownets_8x1_sfine_flyingthings3d_subset_384x768.pth'  # noqa
model = dict(
    type='MaskFlowNet',
    maskflownetS=dict(
        type='MaskFlowNetS',
        freeze_net=True,
        encoder=dict(
            type='PWCNetEncoder',
            in_channels=3,
            net_type='Basic',
            pyramid_levels=[
                'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
            ],
            out_channels=(16, 32, 64, 96, 128, 196),
            strides=(2, 2, 2, 2, 2, 2),
            dilations=(1, 1, 1, 1, 1, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        decoder=dict(
            type='MaskFlowNetSDecoder',
            warp_in_channels=dict(
                level6=196, level5=128, level4=96, level3=64, level2=32),
            up_channels=dict(
                level6=16, level5=16, level4=16, level3=16, level2=16),
            warp_type='AsymOFMM',
            in_channels=dict(
                level6=81, level5=227, level4=195, level3=163, level2=131),
            corr_cfg=dict(type='Correlation', max_displacement=4),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            scaled=False,
            post_processor=dict(type='ContextNet', in_channels=579)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(),
        init_cfg=dict(type='Pretrained', checkpoint=MaskflownetS_checkpoint)),
    encoder=dict(
        type='PWCNetEncoder',
        in_channels=4,
        net_type='Basic',
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        out_channels=(16, 32, 64, 96, 128, 196),
        strides=(2, 2, 2, 2, 2, 2),
        dilations=(1, 1, 1, 1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    decoder=dict(
        type='MaskFlowNetDecoder',
        warp_in_channels=dict(
            level6=196, level5=128, level4=96, level3=64, level2=32),
        up_channels=dict(
            level6=16, level5=16, level4=16, level3=16, level2=16),
        warp_type='Basic',
        with_mask=False,
        in_channels=dict(
            level6=52, level5=198, level4=166, level3=134, level2=102),
        corr_cfg=dict(type='Correlation', max_displacement=2),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled=False,
        post_processor=dict(type='ContextNet', in_channels=550),
        flow_loss=dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights={
                'level2': 0.005,
                'level3': 0.01,
                'level4': 0.02,
                'level5': 0.08,
                'level6': 0.32
            }),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
    init_cfg=dict(
        type='Kaiming', a=0.1, distribution='uniform', layer='Conv2d'))
