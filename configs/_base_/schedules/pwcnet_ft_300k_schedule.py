# optimizer
optimizer = dict(type='Adam', lr=3e-5, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='MultiStage',
    by_epoch=False,
    gammas=[0.5, 0.5],
    milestone_lrs=[3e-5, 2e-5],
    milestone_iters=[0, 150000],
    steps=[
        [
            45000, 65000, 85000, 95000, 97500, 100000, 110000, 120000, 130000,
            140000
        ],
        [
            195000, 215000, 235000, 245000, 247500, 250000, 260000, 270000,
            280000, 290000
        ],
    ])
runner = dict(type='IterBasedRunner', max_iters=300000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')
