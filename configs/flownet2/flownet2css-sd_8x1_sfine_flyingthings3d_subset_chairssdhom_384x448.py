_base_ = [
    '../_base_/models/flownet2/flownet2css-sd.py',
    '../_base_/datasets/flyingthings3d_subset_chairssdhom_384x448.py',
    '../_base_/schedules/schedule_s_fine.py', '../_base_/default_runtime.py'
]

# Initialize FlowNet2CSS and finetune on FlingThings3D_subset and ChairsSDHom
load_from = 'https://download.openmmlab.com/mmflow/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'  # noqa
