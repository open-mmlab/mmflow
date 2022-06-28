# training schedule for S_short schedule
train_cfg = dict(by_epoch=False, max_iters=600000, val_interval=50000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001, weight_decay=0.0004, betas=(0.9, 0.999)))

# learning policy
lr_config = dict(
    type='MultiStepLR',
    by_epoch=False,
    gamma=0.5,
    milestones=[300000, 400000, 500000])
