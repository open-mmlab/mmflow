_base_ = [
    '../_base_/models/flownet2/flownet2cs.py',
    '../_base_/datasets/flyingthings3d_subset_384x768.py',
    '../_base_/schedules/schedule_s_fine.py', '../_base_/default_runtime.py'
]

# Train on FlyingChairs and finetune on FlyingThings3D subset
load_from = 'https://download.openmmlab.com/mmflow/flownet2/flownet2cs_8x1_slong_flyingchairs_384x448.pth'  # noqa
