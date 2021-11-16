# optimizer
optimizer = dict(type='Adam', lr=5e-5, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='MultiStage',
    by_epoch=False,
    gammas=[0.5, 0.5, 0.5, 0.5, 0.5],
    milestone_lrs=[5e-5, 3e-5, 2e-5, 1e-5, 5e-6],
    milestone_iters=[0, 200000, 400000, 600000, 800000],
    steps=[[100000, 150000], [300000, 350000], [500000, 550000],
           [700000, 750000], [850000, 875000, 900000, 950000, 975000]])

runner = dict(type='IterBasedRunner', max_iters=1000000)
checkpoint_config = dict(by_epoch=False, interval=100000)
evaluation = dict(interval=100000, metric='EPE')
