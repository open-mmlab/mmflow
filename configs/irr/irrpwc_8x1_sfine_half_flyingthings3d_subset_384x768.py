_base_ = [
    '../_base_/models/irrpwc.py',
    '../_base_/datasets/flyingthings3d_subset_bi_with_occ_384x768.py',
    '../_base_/schedules/schedule_s_fine_half.py',
    '../_base_/default_runtime.py'
]

custom_hooks = [dict(type='EMAHook')]

data = dict(
    train_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=5, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False))

# Train on FlyingChairsOcc and finetune on FlyingThings3D_subset
load_from = 'https://download.openmmlab.com/mmflow/irr/irrpwc_8x1_sshort_flyingchairsocc_384x448.pth'  # noqa
