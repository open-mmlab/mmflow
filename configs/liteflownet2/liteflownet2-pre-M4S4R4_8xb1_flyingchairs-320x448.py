_base_ = [
    '../_base_/models/liteflownet2/liteflownet2_pre_M4S4R4.py',
    '../_base_/datasets/flyingchairs_320x448.py',
    '../_base_/schedules/liteflownet_pre_240k.py',
    '../_base_/default_runtime.py'
]

# Weights are initialized from model of previous stage
load_from = 'https://download.openmmlab.com/mmflow/liteflownet2/liteflownet2_pre_M5S5R5_8x1_flyingchairs_320x448.pth'  # noqa
