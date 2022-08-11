# training schedule for pwc-net_ft schedule
train_cfg = dict(by_epoch=False, max_iters=300000, val_interval=50000)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=3e-5, weight_decay=0.0004, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='MultiStageLR',
    by_epoch=False,
    gammas=[0.5, 0.5],
    milestone_params=[3e-5, 2e-5],
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

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50000, by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='FlowVisualizationHook'))
