_base_ = [
    '../_base_/models/gma/gma.py',
    '../_base_/datasets/kitti2015_raft_288x960.py',
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
    total_steps=50100,
    pct_start=0.05,
    anneal_strategy='linear')

runner = dict(type='IterBasedRunner', max_iters=50000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=5000, metric='EPE')

# Load model training on mixed datasets and finetune it on KITTI2015s
load_from = 'https://download.openmmlab.com/mmflow/gma/ gma_8x2_120k_mixed_368x768.pth'  # noqa
