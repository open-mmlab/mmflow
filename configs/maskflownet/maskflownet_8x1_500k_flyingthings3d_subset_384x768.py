_base_ = [
    '../_base_/models/maskflownet.py',
    '../_base_/datasets/flyingthings3d_subset_384x768.py',
    '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(
    type='Adam', lr=1.0e-5, weight_decay=4.0e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step', by_epoch=False, gamma=0.5, step=[200000, 300000, 400000])
runner = dict(type='IterBasedRunner', max_iters=500000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')

# Train on FlyingChairs and finetune on FlyingThings3D_subset
load_from = 'https://download.openmmlab.com/mmflow/maskflownet/maskflownet_8x1_800k_flyingchairs_384x448.pth'  # noqa
