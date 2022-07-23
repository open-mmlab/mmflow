test_dataset_type = 'Sintel'
test_data_root = 'data/Sintel'

img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False)

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
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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
    dict(type='RandomCrop', crop_size=(384, 768)),
    dict(type='PackFlowInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputResize', exponent=6),
    dict(type='PackFlowInputs')
]

flyingthings3d_subset_train = dict(
    type='FlyingThings3DSubset',
    pipeline=train_pipeline,
    data_root='data/FlyingThings3D_subset',
    test_mode=False,
    scene='left')

test_data_cleanpass = dict(
    type=test_dataset_type,
    data_root=test_data_root,
    pipeline=test_pipeline,
    test_mode=False,
    pass_style='clean')

test_data_finalpass = dict(
    type=test_dataset_type,
    data_root=test_data_root,
    pipeline=test_pipeline,
    test_mode=False,
    pass_style='final')

train_dataloader = dict(
    batch_size=1,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    num_workers=2,
    drop_last=True,
    persistent_workers=True,
    dataset=flyingthings3d_subset_train)

val_dataloader = [
    dict(
        batch_size=1,
        num_workers=2,
        sampler=dict(type='DefaultSampler', shuffle=False),
        drop_last=False,
        persistent_workers=True,
        dataset=test_data_cleanpass),
    dict(
        batch_size=1,
        num_workers=2,
        sampler=dict(type='DefaultSampler', shuffle=False),
        drop_last=False,
        persistent_workers=True,
        dataset=test_data_finalpass)
]
test_dataloader = val_dataloader
val_evaluator = [
    dict(type='EndPointError', prefix='clean'),
    dict(type='EndPointError', prefix='final')
]
test_evaluator = val_evaluator
