_base_ = [
    '../_base_/models/liteflownet/liteflownet.py',
    '../_base_/datasets/sintel_pwcnet_384x768.py',
    '../_base_/schedules/liteflownet_ft_500k.py',
    '../_base_/default_runtime.py'
]

model = dict(
    decoder=dict(
        flow_loss=dict(
            _delete_=True,
            type='MultiLevelCharbonnierLoss',
            weights=dict(
                level6=0.32,
                level5=0.08,
                level4=0.02,
                level3=0.01,
                level2=0.005),
            q=0.2,
            eps=0.01,
            reduction='sum'),
        init_cfg=None))

# Train on FlyingChairs and FlyingThings3D_subset and finetune on Sintel
load_from = 'https://download.openmmlab.com/mmflow/liteflownet/liteflownet_8x1_500k_flyingthings3d_subset_384x768.pth'  # noqa
