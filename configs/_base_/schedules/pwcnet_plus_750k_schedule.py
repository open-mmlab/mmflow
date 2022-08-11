# training schedule for pwc-net+ schedule
train_cfg = dict(by_epoch=False, max_iters=750000, val_interval=50000)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=5e-5, weight_decay=0.0004, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='MultiStageLR',
    by_epoch=False,
    gammas=[0.5, 0.5, 0.5, 0.5, 0.5],
    milestone_params=[5e-5, 3e-5, 2e-5, 1e-5, 5e-6],
    milestone_iters=[0, 150000, 300000, 450000, 600000],
    steps=[[
        45000, 65000, 85000, 95000, 97500, 100000, 110000, 120000, 130000,
        140000
    ],
           [
               195000, 215000, 235000, 245000, 247500, 250000, 260000, 270000,
               280000, 290000
           ],
           [
               345000, 365000, 385000, 395000, 397500, 400000, 410000, 420000,
               430000, 440000
           ],
           [
               495000, 515000, 535000, 545000, 547500, 550000, 560000, 570000,
               580000, 590000
           ],
           [
               645000, 665000, 685000, 695000, 697500, 700000, 710000, 720000,
               730000, 740000
           ]])

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50000, by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='FlowVisualizationHook'))
