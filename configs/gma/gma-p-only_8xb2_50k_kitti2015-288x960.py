_base_ = [
    '../_base_/models/gma/gma_p-only.py',
    '../_base_/datasets/kitti2015_raft_288x960.py',
    '../_base_/schedules/raft_50k.py', '../_base_/default_runtime.py'
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
        position_only=True,
        max_pos_size=160,
        flow_loss=dict(type='SequenceLoss', gamma=0.85),
        act_cfg=dict(type='ReLU')),
    freeze_bn=False,
    test_cfg=dict(iters=32))

# Load model training on mixed datasets and finetune it on KITTI2015s
load_from = 'https://download.openmmlab.com/mmflow/gma/gma_p-only_8x2_120k_mixed_368x768.pth'  # noqa
