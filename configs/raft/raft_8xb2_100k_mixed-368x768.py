_base_ = [
    '../_base_/models/raft.py',
    '../_base_/datasets/sintel_cleanx100_sintel_fianlx100_kitti2015x200_hd1kx5_flyingthings3d_raft_384x768.py',  # noqa
    '../_base_/schedules/raft_100k.py',
    '../_base_/default_runtime.py'
]

model = dict(
    decoder=dict(
        type='RAFTDecoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=12,
        corr_op_cfg=dict(type='CorrLookup', align_corners=True),
        gru_type='SeqConv',
        flow_loss=dict(type='SequenceLoss', gamma=0.85),
        act_cfg=dict(type='ReLU')),
    freeze_bn=True,
    test_cfg=dict(iters=32))

optim_wrapper = dict(optimizer=dict(lr=0.000125, weight_decay=0.00001))
param_scheduler = dict(eta_max=0.000125)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')
# Train on FlyingChairs and FlyingThings3D, and finetune on
# and Sintel, KITTI2015 and HD1K
load_from = 'https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_flyingthings3d_400x720.pth'  # noqa
