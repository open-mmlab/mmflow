_base_ = [
    '../_base_/models/irrpwc.py',
    '../_base_/datasets/flyingchairsocc_bi_occ_384x448.py',
    '../_base_/schedules/schedule_s_short.py', '../_base_/default_runtime.py'
]

data = dict(
    train_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=5, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False))

custom_hooks = [dict(type='EMAHook')]
