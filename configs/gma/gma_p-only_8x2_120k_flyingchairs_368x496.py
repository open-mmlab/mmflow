_base_ = [
    '../_base_/models/gma/gma_p-only.py',
    '../_base_/datasets/flyingchairs_raft_368x496.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(
    type='AdamW',
    lr=0.00025,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0001,
    amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.))
lr_config = dict(
    policy='OneCycle',
    max_lr=0.00025,
    total_steps=120100,
    pct_start=0.05,
    anneal_strategy='linear')

runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='EPE')
