_base_ = [
    '../_base_/models/maskflownet.py',
    '../_base_/datasets/flyingchairs_384x448.py',
    '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(
    type='Adam', lr=1.0e-4, weight_decay=4.0e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[300000, 500000, 600000, 700000])
runner = dict(type='IterBasedRunner', max_iters=800000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')
