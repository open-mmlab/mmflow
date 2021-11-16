FlowNet2css_checkpoint = 'https://download.openmmlab.com/mmflow/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'  # noqa
FlowNet2sd_checkpoint = 'https://download.openmmlab.com/mmflow/flownet2/flownet2sd_8x1_slong_chairssdhom_384x448.pth'  # noqa: E501

model = dict(
    type='FlowNet2',
    flownetCSS=dict(
        type='FlowNetCSS',
        freeze_net=True,
        flownetC=dict(
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
                    level6=1024,
                    level5=1026,
                    level4=770,
                    level3=386,
                    level2=194),
                out_channels=dict(
                    level6=512, level5=256, level4=128, level3=64),
                deconv_bias=True,
                pred_bias=True,
                upsample_bias=True,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict()),
        flownetS1=dict(
            type='FlowNetS',
            encoder=dict(
                type='FlowNetEncoder',
                in_channels=12,
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
                    level6=1024,
                    level5=1026,
                    level4=770,
                    level3=386,
                    level2=194),
                out_channels=dict(
                    level6=512, level5=256, level4=128, level3=64),
                deconv_bias=True,
                pred_bias=True,
                upsample_bias=False,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict()),
        flownetS2=dict(
            type='FlowNetS',
            encoder=dict(
                type='FlowNetEncoder',
                in_channels=12,
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
                    level6=1024,
                    level5=1026,
                    level4=770,
                    level3=386,
                    level2=194),
                out_channels=dict(
                    level6=512, level5=256, level4=128, level3=64),
                deconv_bias=True,
                pred_bias=True,
                upsample_bias=False,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(),
        ),
        link_cfg=dict(scale_factor=4, mode='bilinear'),
        out_level='level2',
        flow_div=20.,
        init_cfg=dict(type='Pretrained', checkpoint=FlowNet2css_checkpoint),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(),
    ),
    flownetSD=dict(
        type='FlowNetS',
        freeze_net=True,
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
        ),
        init_cfg=dict(type='Pretrained', checkpoint=FlowNet2sd_checkpoint),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(),
    ),
    flownet_fusion=dict(
        type='FlowNetS',
        encoder=dict(
            type='FlowNetEncoder',
            in_channels=11,
            pyramid_levels=['level1', 'level2', 'level3'],
            num_convs=(1, 2, 2),
            out_channels=(64, (64, 128), 128),
            kernel_size=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        ),
        decoder=dict(
            type='FlowNetSDecoder',
            in_channels=dict(level3=128, level2=162, level1=82),
            out_channels=dict(level3=32, level2=16),
            inter_channels=dict(level2=32, level1=16),
            deconv_bias=True,
            pred_bias=True,
            upsample_bias=True,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            flow_loss=dict(
                type='MultiLevelEPE',
                p=2,
                reduction='sum',
                flow_div=1.,
                weights=dict(level1=0.005)),
            flow_div=1.,
            init_cfg=[
                dict(
                    type='Xavier',
                    distribution='uniform',
                    layer=['Conv2d', 'ConvTranspose2d'],
                    bias=0),
                dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)
            ])),
    link_cfg=dict(scale_factor=4, mode='nearest'),
    flow_div=20.,
    out_level='level2',
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict())
