model = dict(
    type='MaskFlowNetS',
    freeze_net=False,
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
        post_processor=dict(type='ContextNet', in_channels=579),
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
