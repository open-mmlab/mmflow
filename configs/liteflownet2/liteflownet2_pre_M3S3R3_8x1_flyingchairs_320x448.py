_base_ = [
    '../_base_/models/liteflownet2/liteflownet2_pre_M3S3R3.py',
    '../_base_/datasets/flyingchairs_320x448.py',
    '../_base_/schedules/liteflownet_pre_240k.py',
    '../_base_/default_runtime.py'
]

optim_wrapper = dict(
    optimizer=dict(
        type='Adam', lr=6e-5, weight_decay=0.0004, betas=(0.9, 0.999)),
    clip_grad=None)

# Weights are initialized from model of previous stage
load_from = 'https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M4S4R4_8x1_flyingchairs_320x448.pth'  # noqa
