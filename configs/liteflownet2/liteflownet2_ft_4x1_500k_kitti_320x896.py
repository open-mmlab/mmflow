_base_ = [
    '../_base_/datasets/kitti2012_kitti2015_320x896.py',
    '../_base_/schedules/liteflownet_ft_500k.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='LiteFlowNet',
    encoder=dict(
        type='NetC',
        in_channels=3,
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        out_channels=(32, 32, 64, 96, 128, 192),
        strides=(1, 2, 2, 2, 2, 2),
        num_convs=(1, 3, 2, 2, 1, 1),
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=None),
    decoder=dict(
        type='NetE',
        in_channels=dict(level3=64, level4=96, level5=128, level6=192),
        corr_channels=dict(level3=49, level4=49, level5=49, level6=49),
        sin_channels=dict(level3=130, level4=194, level5=258, level6=386),
        rin_channels=dict(level3=131, level4=131, level5=131, level6=195),
        feat_channels=64,
        mfeat_channels=(128, 128, 96, 64, 32),
        sfeat_channels=(128, 128, 96, 64, 32),
        rfeat_channels=(128, 128, 64, 64, 32, 32),
        patch_size=dict(level3=5, level4=5, level5=3, level6=3),
        corr_cfg=dict(
            level3=dict(
                type='Correlation',
                max_displacement=3,
                stride=2,
                dilation_patch=2),
            level4=dict(type='Correlation', max_displacement=3),
            level5=dict(type='Correlation', max_displacement=3),
            level6=dict(type='Correlation', max_displacement=3)),
        warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        flow_div=20.,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled_corr=False,
        regularized_flow=True,
        extra_training_loss=True,
        flow_loss=dict(
            type='MultiLevelCharbonnierLoss',
            resize_flow='upsample',
            weights=dict(
                level6=0.32,
                level5=0.08,
                level4=0.02,
                level3=0.01,
                level0=6.25e-4),
            q=0.2,
            eps=0.01,
            reduction='sum'),
        init_cfg=None),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
)

# Train on FlyingChairs and FlyingThings3D_subset, and finetune on KITTI
load_from = 'https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.pth'  # noqa
