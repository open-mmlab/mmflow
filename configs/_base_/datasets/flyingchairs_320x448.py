dataset_type = 'FlyingChairs'
data_root = 'data/FlyingChairs_release'

global_transform = dict(
    translates=(0.05, 0.05),
    zoom=(1.0, 1.5),
    shear=(0.86, 1.16),
    rotate=(-10., 10.))

relative_transform = dict(
    translates=(0.00375, 0.00375),
    zoom=(0.985, 1.015),
    shear=(1.0, 1.0),
    rotate=(-1.0, 1.0))

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
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=global_transform,
        relative_transform=relative_transform),
    dict(type='RandomCrop', crop_size=(320, 448)),
    dict(type='PackFlowInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', file_client_args=file_client_args),
    dict(type='InputResize', exponent=6),
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
    batch_size=1,
    num_workers=2,
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
