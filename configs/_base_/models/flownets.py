model = dict(
    type='FlowNetS',
    encoder=dict(
        type='FlowNetEncoder',
        in_channels=6,
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        num_convs=(1, 1, 2, 2, 2, 2),
        out_channels=(64, 128, 256, 512, 512, 1024),
        kernel_size=(7, 5, (5, 3), 3, 3, 3),
        strides=(2, 2, 2, 2, 2, 2),
        dilations=(1, 1, 1, 1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    ),
    decoder=dict(
        type='FlowNetSDecoder',
        in_channels=dict(
            level6=1024, level5=1026, level4=770, level3=386, level2=194),
        out_channels=dict(level6=512, level5=256, level4=128, level3=64),
        deconv_bias=True,
        pred_bias=True,
        upsample_bias=False,
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
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    init_cfg=[
        dict(
            type='Kaiming',
            layer=['Conv2d', 'ConvTranspose2d'],
            a=0.1,
            mode='fan_in',
            nonlinearity='leaky_relu',
            bias=0),
        dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict())
