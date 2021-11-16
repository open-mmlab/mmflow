# optimizer
optimizer = dict(
    type='Adam', lr=0.0001, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step', by_epoch=False, gamma=0.5, step=[300000, 400000, 500000])
runner = dict(type='IterBasedRunner', max_iters=600000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')
