_base_ = [
    '../_base_/models/gma/gma_plus-p.py',
    '../_base_/datasets/flyingthings3d_raft_400x720.py',
    '../_base_/schedules/gma_120k.py', '../_base_/default_runtime.py'
]

model = dict(freeze_bn=False, test_cfg=dict(iters=32))

lr = 0.000125
optim_wrapper = dict(optimizer=dict(lr=lr))
param_scheduler = dict(eta_max=lr)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# Train on FlyingChairs and finetune on FlyingThings3D
load_from = 'https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingchairs_368x496.pth'  # noqa
