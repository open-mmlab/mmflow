_base_ = [
    '../_base_/models/raft.py', '../_base_/datasets/kitti2015_raft_288x960.py',
    '../_base_/schedules/raft_50k.py', '../_base_/default_runtime.py'
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

# Load model training on mixed datasets and finetune it on KITTI2015
load_from = 'https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_mixed_368x768.pth'  # noqa
