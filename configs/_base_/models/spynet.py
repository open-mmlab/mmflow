model = dict(
    type='SpyNet',
    img_channels=3,
    pyramid_levels=[
        'level0', 'level1', 'level2', 'level3', 'level4', 'level5'
    ],
    decoder=dict(
        type='SpyNetDecoder',
        in_channels=8,
        pyramid_levels=[
            'level0', 'level1', 'level2', 'level3', 'level4', 'level5'
        ],
        out_channels=(32, 64, 32, 16, 2),
        kernel_size=7,
        stride=1,
        warp_cfg=dict(type='Warp', align_corners=True),
        act_cfg=dict(type='ReLU'),
    ))
