_base_ = [
    '../_base_/models/irrpwc.py',
    '../_base_/datasets/flyingchairsocc_bi_occ_384x448.py',
    '../_base_/schedules/schedule_s_short.py', '../_base_/default_runtime.py'
]

custom_hooks = [dict(type='EMAHook')]
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
