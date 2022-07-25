train_dataset_type = 'FlyingThings3D'
train_data_root = 'data/flyingthings3d'
test_dataset_type = 'Sintel'
test_data_root = 'data/Sintel'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
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
        crop_size=(400, 720),
        min_scale=-0.4,
        max_scale=0.8,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=(400, 720)),
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

train_dataset_cleanpass = dict(
    type=train_dataset_type,
    data_root=train_data_root,
    pipeline=train_pipeline,
    test_mode=False,
    pass_style='clean',
    scene='left',
    double=True)

train_dataset_finalpass = dict(
    type=train_dataset_type,
    data_root=train_data_root,
    pipeline=train_pipeline,
    test_mode=False,
    pass_style='final',
    scene='left',
    double=True)

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
    batch_size=2,
    num_workers=5,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    drop_last=True,
    persistent_workers=True,
    dataset=dict(
        type='ConcatDataset',
        datasets=[train_dataset_cleanpass, train_dataset_finalpass]))

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
