_base_ = [
    '../_base_/models/raft.py',
    '../_base_/datasets/flyingthings3d_raft_400x720.py',
    '../_base_/default_runtime.py'
]

model = dict(freeze_bn=True, test_cfg=dict(iters=32))

optimizer = dict(
    type='AdamW',
    lr=0.000125,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.00001,
    amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.))
lr_config = dict(
    policy='OneCycle',
    max_lr=0.000125,
    total_steps=100100,
    pct_start=0.05,
    anneal_strategy='linear')

runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='EPE')

# Train on FlyingChairs and finetune on FlyingThings3D
load_from = 'https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_flyingchairs.pth'  # noqa
