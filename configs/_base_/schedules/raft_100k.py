train_cfg = dict(by_epoch=False, max_iters=100000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0004,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0001,
        amsgrad=False),
    clip_grad=dict(max_norm=1.))

# learning policy
param_scheduler = dict(
    type='OneCycleLR',
    eta_max=0.0004,
    total_steps=100100,
    pct_start=0.05,
    anneal_strategy='linear',
    by_epoch=False)
