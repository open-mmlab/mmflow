_base_ = [
    '../_base_/models/raft.py',
    '../_base_/datasets/flyingchairs_raft_368x496.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0001,
    amsgrad=False)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# toy learning policy
lr_config = dict(
    type='MultiStepLR',
    by_epoch=False,
    gamma=0.5,
    milestones=[400000, 600000, 800000, 1000000])
# lr_config = dict(
#     policy='OneCycle',
#     max_lr=0.0004,
#     total_steps=100100,
#     pct_start=0.05,
#     anneal_strategy='linear')

train_cfg = dict(by_epoch=False, max_iters=100000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    optimizer=dict(
        type='OptimizerHook', grad_clip=dict(max_norm=1.), _delete_=True),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
custom_hooks = [dict(type='SyncBuffersHook')]
