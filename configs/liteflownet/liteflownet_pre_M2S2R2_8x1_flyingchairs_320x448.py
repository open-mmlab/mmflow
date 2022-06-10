_base_ = [
    '../_base_/models/liteflownet/liteflownet_pre_M2S2R2.py',
    '../_base_/datasets/flyingchairs_320x448.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(type='Adam', lr=4e-5, weight_decay=0.0004, betas=(0.9, 0.999))
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning policy
lr_config = dict(
    type='MultiStepLR',
    by_epoch=False,
    gamma=0.5,
    step=[120000, 160000, 200000, 240000])
train_cfg = dict(by_epoch=False, max_iters=300000, val_interval=50000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# custom_hooks = [
#     dict(
#         type='LiteFlowNetStageLoadHook',
#         src_level='level3',
#         dst_level='level2')
# ]

# Weights are initialized from model of previous stage
# load_from = 'https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M3S3R3_8x1_flyingchairs_320x448.pth'  # noqa
default_hooks = dict(
    optimizer=dict(type='OptimizerHook', grad_clip=None),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
