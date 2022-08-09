train_cfg = dict(by_epoch=False, max_iters=120000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00025,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0001,
        amsgrad=False),
    clip_grad=dict(max_norm=1.))

# learning policy
param_scheduler = dict(
    type='OneCycleLR',
    eta_max=0.00025,
    total_steps=120100,
    pct_start=0.05,
    anneal_strategy='linear',
    by_epoch=False)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10000, by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='FlowVisualizationHook'))
