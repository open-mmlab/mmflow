_base_ = [
    '../_base_/datasets/kitti2012_kitti2015_irr_320x896.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/pwcnet_ft_300k_schedule.py'
]

model = dict(
    type='IRRPWC',
    data_preprocessor=dict(
        type='FlowDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=False,
        sigma_range=(0, 0.04),
        clamp_range=(0., 1.)),
    encoder=dict(
        type='PWCNetEncoder',
        in_channels=3,
        net_type='Small',
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        out_channels=(16, 32, 64, 96, 128, 196),
        strides=(2, 2, 2, 2, 2, 2),
        dilations=(1, 1, 1, 1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    decoder=dict(
        type='IRRPWCDecoder',
        flow_levels=[
            'level0', 'level1', 'level2', 'level3', 'level4', 'level5',
            'level6'
        ],
        corr_in_channels=dict(
            level2=32, level3=64, level4=96, level5=128, level6=196),
        corr_feat_channels=32,
        flow_decoder_in_channels=115,
        occ_decoder_in_channels=114,
        corr_cfg=dict(type='Correlation', max_displacement=4),
        scaled=True,
        warp_cfg=dict(type='Warp', align_corners=True),
        densefeat_channels=(128, 128, 96, 64, 32),
        flow_post_processor=dict(
            type='ContextNet',
            in_channels=565,
            out_channels=2,
            feat_channels=(128, 128, 128, 96, 64, 32),
            dilations=(1, 2, 4, 8, 16, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        flow_refine=dict(
            type='FlowRefine',
            in_channels=35,
            feat_channels=(128, 128, 64, 64, 32, 32),
            patch_size=3,
            warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        ),
        occ_post_processor=dict(
            type='ContextNet',
            in_channels=563,
            out_channels=1,
            feat_channels=(128, 128, 128, 96, 64, 32),
            dilations=(1, 2, 4, 8, 16, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        occ_refine=dict(
            type='OccRefine',
            in_channels=65,
            feat_channels=(128, 128, 64, 64, 32, 32),
            patch_size=3,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            warp_cfg=dict(type='Warp', align_corners=True),
        ),
        occ_upsample=dict(
            type='OccShuffleUpsample',
            in_channels=11,
            feat_channels=32,
            infeat_channels=16,
            out_channels=1,
            warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        ),
        occ_refined_levels=['level0', 'level1'],
        flow_div=20.,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        flow_loss=dict(
            type='MultiLevelEPE',
            weights=dict(
                level6=0.32,
                level5=0.08,
                level4=0.02,
                level3=0.01,
                level2=0.005,
                level1=0.00125,
                level0=0.0003125),
            p=1,
            q=0.4,
            eps=0.01,
            resize_flow='upsample',
            reduction='sum'),
    ),
    init_cfg=dict(
        type='Kaiming',
        a=0.1,
        nonlinearity='leaky_relu',
        layer=['Conv2d', 'ConvTranspose2d'],
        mode='fan_in',
        bias=0),
    train_cfg=dict(),
    test_cfg=dict())

custom_hooks = [dict(type='EMAHook')]

# kitti dataset don't include occlusion.
find_unused_parameters = True

# Train on FlyingChairsOcc and FlyingThings3D_subset, and finetune on KITTI
load_from = 'https://download.openmmlab.com/mmflow/irr/irrpwc_8x1_sfine_half_flyingthings3d_subset_384x768.pth'  # noqa
