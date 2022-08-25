crop_size = (320, 896)

global_transform = dict(
    translates=(0.02, 0.02),
    zoom=(0.98, 1.02),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))

relative_transform = dict(
    translates=(0.0025, 0.0025),
    zoom=(0.99, 1.01),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))

sparse_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(
        type='ColorJitter',
        brightness=0.05,
        contrast=0.2,
        saturation=0.25,
        hue=0.1),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=global_transform,
        relative_transform=relative_transform),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='PackFlowInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(type='InputResize', exponent=6),
    dict(type='PackFlowInputs')
]

kitti2015_train = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=sparse_train_pipeline,
    test_mode=False)

kitti2015_val_test = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=test_pipeline,
    test_mode=False)

kitti2012_train = dict(
    type='KITTI2012',
    data_root='data/kitti2012',
    pipeline=sparse_train_pipeline,
    test_mode=False)

kitti2012_val_test = dict(
    type='KITTI2012',
    data_root='data/kitti2012',
    pipeline=test_pipeline,
    test_mode=False)

train_dataloader = dict(
    batch_size=1,
    num_workers=5,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    drop_last=True,
    persistent_workers=True,
    dataset=dict(
        type='ConcatDataset', datasets=[kitti2015_train, kitti2012_train]))

val_dataloader = [
    dict(
        batch_size=1,
        num_workers=2,
        sampler=dict(type='DefaultSampler', shuffle=False),
        drop_last=False,
        persistent_workers=True,
        dataset=kitti2015_val_test),
    dict(
        batch_size=1,
        num_workers=2,
        sampler=dict(type='DefaultSampler', shuffle=False),
        drop_last=False,
        persistent_workers=True,
        dataset=kitti2012_val_test)
]
test_dataloader = val_dataloader

val_evaluator = [
    [
        dict(type='EndPointError', prefix='KITTI2015'),
        dict(type='FlowOutliers', prefix='KITTI2015')
    ],
    [
        dict(type='EndPointError', prefix='KITTI2012'),
        dict(type='FlowOutliers', prefix='KITTI2012')
    ],
]
test_evaluator = val_evaluator
