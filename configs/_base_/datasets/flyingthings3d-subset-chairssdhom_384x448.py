test_dataset_type = 'Sintel'
test_data_root = 'data/Sintel'

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
    dict(type='RandomCrop', crop_size=(384, 448)),
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

chairssdHom_train = dict(
    type='ChairsSDHom',
    pipeline=train_pipeline,
    data_root='data/ChairsSDHom',
    test_mode=False)

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
    sampler=dict(
        type='MixedBatchDistributedSampler',
        sample_ratio=[0.25, 0.75],
        shuffle=True),
    num_workers=5,
    drop_last=True,
    persistent_workers=True,
    dataset=dict(
        type='ConcatDataset',
        datasets=[flyingthings3d_subset_train, chairssdHom_train]))

val_dataloader = [
    dict(
        batch_size=1,
        num_workers=5,
        sampler=dict(type='DefaultSampler', shuffle=False),
        drop_last=False,
        persistent_workers=True,
        dataset=test_data_cleanpass),
    dict(
        batch_size=1,
        num_workers=5,
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
