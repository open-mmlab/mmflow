dataset_type = 'FlyingChairsOcc'
data_root = 'data/FlyingChairsOcc/'

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

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_occ=True),
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
        relative_transform=relative_transform,
        check_bound=True),
    dict(type='RandomCrop', crop_size=(384, 448)),
    dict(type='PackFlowInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputResize', exponent=6),
    dict(type='PackFlowInputs')
]

flyingchairsocc_train = dict(
    type=dataset_type,
    data_root=data_root,
    pipeline=train_pipeline,
    test_mode=False)

flyingchairsocc_test = dict(
    type=dataset_type,
    data_root=data_root,
    pipeline=test_pipeline,
    test_mode=True)

train_dataloader = dict(
    batch_size=1,
    num_workers=5,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    drop_last=True,
    persistent_workers=True,
    dataset=flyingchairsocc_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=5,
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=False,
    persistent_workers=True,
    dataset=flyingchairsocc_test)

test_dataloader = val_dataloader
val_evaluator = dict(type='EndPointError')
test_evaluator = val_evaluator
