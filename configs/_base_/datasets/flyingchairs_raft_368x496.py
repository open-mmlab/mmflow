dataset_type = 'FlyingChairs'
data_root = 'data/FlyingChairs_release'
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/mmflow/',
#         'data/': 's3://openmmlab/datasets/mmflow/'
#     }))
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', file_client_args=file_client_args),
    dict(
        type='ColorJitter',
        asymmetric_prob=0.2,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.5 / 3.14),
    dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
    dict(
        type='SpacialTransform',
        spacial_prob=0.8,
        stretch_prob=0.8,
        crop_size=(368, 496),
        min_scale=-0.1,
        max_scale=1.,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=(368, 496)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.1, direction='vertical'),
    dict(type='Validation', max_flow=1000.),
    dict(type='PackFlowInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputPad', exponent=3),
    dict(type='PackFlowInputs')
]

flyingchairs_train = dict(
    type=dataset_type,
    pipeline=train_pipeline,
    data_root=data_root,
    split_file='data/FlyingChairs_release/FlyingChairs_train_val.txt')
flyingchairs_test = dict(
    type=dataset_type,
    pipeline=test_pipeline,
    data_root=data_root,
    test_mode=True,
    split_file='data/FlyingChairs_release/FlyingChairs_train_val.txt')
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    drop_last=True,
    persistent_workers=True,
    dataset=flyingchairs_train)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=False,
    persistent_workers=True,
    dataset=flyingchairs_test)
test_dataloader = val_dataloader
val_evaluator = dict(type='EndPointError')
test_evaluator = val_evaluator
