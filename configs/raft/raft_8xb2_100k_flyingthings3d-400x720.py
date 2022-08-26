_base_ = [
    '../_base_/models/raft.py',
    '../_base_/datasets/flyingthings3d_raft_400x720.py',
    '../_base_/schedules/raft_100k.py', '../_base_/default_runtime.py'
]

model = dict(freeze_bn=True, test_cfg=dict(iters=32))

optim_wrapper = dict(optimizer=dict(lr=0.000125, weight_decay=0.00001))
param_scheduler = dict(eta_max=0.000125)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')
# Train on FlyingChairs and finetune on FlyingThings3D
load_from = 'https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_flyingchairs.pth'  # noqa
