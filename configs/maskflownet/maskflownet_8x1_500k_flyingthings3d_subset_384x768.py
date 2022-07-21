_base_ = [
    '../_base_/models/maskflownet.py',
    '../_base_/datasets/flyingthings3d_subset_384x768.py',
    '../_base_/schedules/maskflownet_500k.py', '../_base_/default_runtime.py'
]

# Train on FlyingChairs and finetune on FlyingThings3D_subset
load_from = 'https://download.openmmlab.com/mmflow/maskflownet/maskflownet_8x1_800k_flyingchairs_384x448.pth'  # noqa
