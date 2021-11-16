_base_ = [
    '../_base_/models/flownet2/flownet2css.py',
    '../_base_/datasets/flyingchairs_384x448.py',
    '../_base_/schedules/schedule_s_long.py', '../_base_/default_runtime.py'
]
# Initialize weights of FlowNet2CS with
load_from = 'https://download.openmmlab.com/mmflow/flownet2/flownet2cs_8x1_sfine_flyingthings3d_subset_384x768.pth'  # noqa
