model = dict(
    type='PWCNet',
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
        type='PWCNetDecoder',
        in_channels=dict(
            level6=81, level5=213, level4=181, level3=149, level2=117),
        flow_div=20.,
        corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
        warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled=False,
        post_processor=dict(type='ContextNet', in_channels=565),
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
        type='Kaiming',
        nonlinearity='leaky_relu',
        layer=['Conv2d', 'ConvTranspose2d'],
        mode='fan_in',
        bias=0))
