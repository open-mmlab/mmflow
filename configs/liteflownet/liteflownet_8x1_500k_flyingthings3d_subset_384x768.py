_base_ = [
    '../_base_/models/liteflownet/liteflownet.py',
    '../_base_/datasets/flyingthings3d_subset_384x768.py',
    '../_base_/schedules/liteflownet_ft_500k.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(type='Adam', lr=3e-6, weight_decay=0.0004, betas=(0.9, 0.999))
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# Train on FlyingChairs and finetune on FlyingThings3D_subset
load_from = 'https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M2S2R2_8x1_flyingchairs_320x448.pth'  # noqa
