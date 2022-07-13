_base_ = [
    '../_base_/models/raft.py',
    '../_base_/datasets/flyingchairs_raft_368x496.py',
    '../_base_/schedules/raft_100k.py', '../_base_/default_runtime.py'
]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10000))
