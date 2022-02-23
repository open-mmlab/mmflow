_base_ = [
    '../_base_/models/gma/gma_plus-p.py',
    '../_base_/datasets/sintel_cleanx100_sintel_fianlx100_kitti2015x200_hd1kx5_flyingthings3d_raft_384x768.py',  # noqa
    '../_base_/default_runtime.py'
]

model = dict(
    decoder=dict(
        type='GMADecoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=12,
        corr_op_cfg=dict(type='CorrLookup', align_corners=True),
        gru_type='SeqConv',
        heads=1,
        motion_channels=128,
        position_only=False,
        max_pos_size=160,
        flow_loss=dict(type='SequenceLoss', gamma=0.85),
        act_cfg=dict(type='ReLU')),
    freeze_bn=False,
    test_cfg=dict(iters=32))

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
    total_steps=120100,
    pct_start=0.05,
    anneal_strategy='linear')

runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='EPE')

# Train on FlyingChairs and FlyingThings3D, and finetune on
# and Sintel, KITTI2015 and HD1K
load_from = 'https://download.openmmlab.com/mmflow/gma/gma_plus-p_8x2_120k_flyingthings3d_400x720.pth'  # noqa
