_base_ = [
    '../_base_/models/liteflownet/liteflownet.py',
    '../_base_/datasets/flyingthings3d_subset_384x768.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(type='Adam', lr=3e-6, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step', by_epoch=False, gamma=0.5, step=[200000, 300000, 400000])
runner = dict(type='IterBasedRunner', max_iters=500000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')

# Train on FlyingChairs and finetune on FlyingThings3D_subset
load_from = 'https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M2S2R2_8x1_flyingchairs_320x448.pth'  # noqa
