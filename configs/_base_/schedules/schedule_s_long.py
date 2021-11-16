# optimizer
optimizer = dict(
    type='Adam', lr=0.0001, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[400000, 600000, 800000, 1000000])
runner = dict(type='IterBasedRunner', max_iters=1200000)
checkpoint_config = dict(by_epoch=False, interval=100000)
evaluation = dict(interval=100000, metric='EPE')
