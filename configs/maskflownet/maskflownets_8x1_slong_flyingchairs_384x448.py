_base_ = [
    '../_base_/models/maskflownets.py',
    '../_base_/datasets/flyingchairs_384x448.py',
    '../_base_/schedules/schedule_s_long.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='Adam', lr=0.0001, weight_decay=0., betas=(0.9, 0.999))
