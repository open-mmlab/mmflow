_base_ = [
    '../_base_/models/gma/gma.py',
    '../_base_/datasets/sintel_cleanx100_sintel_fianlx100_flyingthings3d_raft_368x768.py',  # noqa
    '../_base_/schedules/gma_120k.py',
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

lr = 0.000125
optim_wrapper = dict(optimizer=dict(lr=lr))
param_scheduler = dict(eta_max=lr)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# Train on FlyingChairs and FlyingThings3D, and finetune on FlyingThings3D
# and Sintel
load_from = 'https://download.openmmlab.com/mmflow/gma/gma_8x2_120k_flyingthings3d_400x720.pth'  # noqa
