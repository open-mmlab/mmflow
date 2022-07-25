# training schedule for S_short schedule
train_cfg = dict(by_epoch=False, max_iters=300000, val_interval=50000)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.00001, weight_decay=0.0004, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    gamma=0.5,
    milestones=[100000, 150000, 200000, 250000])
