_base_ = [
    '../_base_/models/liteflownet2/liteflownet2.py',
    '../_base_/datasets/sintel_kitti_liteflownet2_320x768.py',
    '../_base_/schedules/liteflownet_ft_600k.py',
    '../_base_/default_runtime.py'
]
model = dict(
    decoder=dict(
        flow_loss=dict(
            _delete_=True,
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
        init_cfg=None))

# Train on FlyingChairs and FlyingThings3D_subset, and finetune on
# Sintel and KITTI
load_from = 'https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_8x1_500k_flyingthing3d_subset_384x768.pth'  # noqa
