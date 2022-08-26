_base_ = [
    '../_base_/models/raft.py', '../_base_/datasets/kitti2015_raft_test.py',
    '../_base_/default_runtime.py'
]
model = dict(freeze_bn=True, test_cfg=dict(iters=32))
