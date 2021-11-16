# optimizer
optimizer = dict(
    type='Adam', lr=0.00001, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[100000, 150000, 200000, 250000])
runner = dict(type='IterBasedRunner', max_iters=300000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')
