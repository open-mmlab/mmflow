model = dict(
    type='FlowNetS',
    data_preprocessor=dict(
        type='FlowDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=False,
        sigma_range=(0, 0.04),
        clamp_range=(0., 1.)),
    encoder=dict(
        type='FlowNetSDEncoder',
        in_channels=6,
        plugin_channels=64,
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        num_convs=(2, 2, 2, 2, 2, 2),
        out_channels=((64, 128), 128, 256, 512, 512, 1024),
        kernel_size=3,
        strides=(2, 2, 2, 2, 2, 2),
        dilations=(1, 1, 1, 1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    ),
    decoder=dict(
        type='FlowNetSDecoder',
        in_channels=dict(
            level6=1024, level5=1026, level4=770, level3=386, level2=194),
        out_channels=dict(level6=512, level5=256, level4=128, level3=64),
        inter_channels=dict(level5=512, level4=256, level3=128, level2=64),
        deconv_bias=True,
        pred_bias=True,
        upsample_bias=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
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
randomness = dict(seed=0, diff_rank_seed=True)
