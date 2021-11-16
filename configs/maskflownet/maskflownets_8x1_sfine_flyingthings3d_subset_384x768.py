_base_ = [
    '../_base_/models/maskflownets.py',
    '../_base_/datasets/flyingthings3d_subset_384x768.py',
    '../_base_/schedules/schedule_s_fine.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='Adam', lr=0.0001, weight_decay=0., betas=(0.9, 0.999))

# Train on FlyingChairs and finetune on FlyingThings3D_subset
load_from = 'https://download.openmmlab.com/mmflow/maskflownet/maskflownets_8x1_slong_flyingchairs_384x448.pth'  # noqa
