# training schedule for S_short schedule
train_cfg = dict(by_epoch=False, max_iters=300000, val_interval=50000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optimizer_config = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.00001, weight_decay=0.0004, betas=(0.9, 0.999)))

# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[100000, 150000, 200000, 250000])
