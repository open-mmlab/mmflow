model = dict(
    type='FlowNetC',
    encoder=dict(
        type='FlowNetEncoder',
        in_channels=3,
        pyramid_levels=['level1', 'level2', 'level3'],
        out_channels=(64, 128, 256),
        kernel_size=(7, 5, 5),
        strides=(2, 2, 2),
        num_convs=(1, 1, 1),
        dilations=(1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    ),
    corr_level='level3',
    corr_encoder=dict(
        type='CorrEncoder',
        in_channels=473,
        pyramid_levels=['level3', 'level4', 'level5', 'level6'],
        kernel_size=(3, 3, 3, 3),
        num_convs=(1, 2, 2, 2),
        out_channels=(256, 512, 512, 1024),
        redir_in_channels=256,
        redir_channels=32,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        corr_cfg=dict(
            type='Correlation',
            kernel_size=1,
            max_displacement=10,
            stride=1,
            padding=0,
            dilation_patch=2),
        scaled=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    ),
    decoder=dict(
        type='FlowNetCDecoder',
        in_channels=dict(
            level6=1024, level5=1026, level4=770, level3=386, level2=194),
        out_channels=dict(level6=512, level5=256, level4=128, level3=64),
        deconv_bias=True,
        pred_bias=True,
        upsample_bias=True,
        norm_cfg=None,
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
    ],  # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict())
