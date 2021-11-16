_base_ = [
    '../_base_/models/flownet2/flownet2css.py',
    '../_base_/datasets/flyingthings3d_subset_384x768.py',
    '../_base_/schedules/schedule_s_fine.py', '../_base_/default_runtime.py'
]

# Train on FlyingChairs and finetune on FlyingThings3D subset
load_from = 'https://download.openmmlab.com/mmflow/flownet2/flownet2css_8x1_sfine_flyingthings3d_subset_384x768.pth'  # noqa
