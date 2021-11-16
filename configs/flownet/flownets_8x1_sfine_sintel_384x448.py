_base_ = [
    '../_base_/models/flownets.py', '../_base_/datasets/sintel_384x768.py',
    '../_base_/schedules/schedule_s_fine.py', '../_base_/default_runtime.py'
]

# Train on FlyingChairs and finetune on Sintel
load_from = 'https://download.openmmlab.com/mmflow/flownet/flownets_8x1_slong_flyingchairs_384x448.pth'  # noqa
