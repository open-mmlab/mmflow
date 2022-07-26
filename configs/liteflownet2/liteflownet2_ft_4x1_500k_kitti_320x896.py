_base_ = [
    '../_base_/models/liteflownet2/liteflownet2.py',
    '../_base_/datasets/kitti2012_kitti2015_320x896.py',
    '../_base_/schedules/liteflownet_ft_500k.py',
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

# Train on FlyingChairs and FlyingThings3D_subset, and finetune on KITTI
load_from = 'https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.pth'  # noqa
