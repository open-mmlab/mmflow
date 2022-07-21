_base_ = [
    '../_base_/models/liteflownet/liteflownet_pre_M2S2R2.py',
    '../_base_/datasets/flyingchairs_320x448.py',
    '../_base_/schedules/liteflownet_pre_300k.py',
    '../_base_/default_runtime.py'
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=4e-5, weight_decay=0.0004, betas=(0.9, 0.999)))

custom_hooks = [
    dict(
        type='LiteFlowNetStageLoadHook',
        src_level='level3',
        dst_level='level2')
]

# Weights are initialized from model of previous stage
load_from = 'https://download.openmmlab.com/mmflow/liteflownet/liteflownet_pre_M3S3R3_8x1_flyingchairs_320x448.pth'  # noqa
