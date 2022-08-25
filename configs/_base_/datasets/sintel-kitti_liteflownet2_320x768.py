crop_size = (320, 768)

sintel_global_transform = dict(
    translates=(0.05, 0.05),
    zoom=(1.0, 1.2),
    shear=(0.95, 1.1),
    rotate=(-5., 5.))

sintel_relative_transform = dict(
    translates=(0.00375, 0.00375),
    zoom=(0.985, 1.015),
    shear=(1.0, 1.0),
    rotate=(-1.0, 1.0))

sintel_train_pipeline = [
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
        global_transform=sintel_global_transform,
        relative_transform=sintel_relative_transform),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='PackFlowInputs')
]

sintel_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputResize', exponent=4),
    dict(type='PackFlowInputs')
]

sintel_clean_train = dict(
    type='Sintel',
    pipeline=sintel_train_pipeline,
    data_root='data/Sintel',
    test_mode=False,
    pass_style='clean')

sintel_final_train = dict(
    type='Sintel',
    pipeline=sintel_train_pipeline,
    data_root='data/Sintel',
    test_mode=False,
    pass_style='final')

sintel_clean_test = dict(
    type='Sintel',
    pipeline=sintel_test_pipeline,
    data_root='data/Sintel',
    test_mode=False,
    pass_style='clean')

sintel_final_test = dict(
    type='Sintel',
    pipeline=sintel_test_pipeline,
    data_root='data/Sintel',
    test_mode=False,
    pass_style='final')

sparse_global_transform = dict(
    translates=(0.02, 0.02),
    zoom=(0.98, 1.02),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))

sparse_relative_transform = dict(
    translates=(0.0025, 0.0025),
    zoom=(0.99, 1.01),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))

sparse_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
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
        global_transform=sparse_global_transform,
        relative_transform=sparse_relative_transform),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='PackFlowInputs')
]

sparse_test_pipeline = [
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
    pipeline=sparse_test_pipeline,
    test_mode=False)

kitti2012_train = dict(
    type='KITTI2012',
    data_root='data/kitti2012',
    pipeline=sparse_train_pipeline,
    test_mode=False)

kitti2012_val_test = dict(
    type='KITTI2012',
    data_root='data/kitti2012',
    pipeline=sparse_test_pipeline,
    test_mode=False)

train_dataloader = dict(
    batch_size=1,
    num_workers=5,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    drop_last=True,
    persistent_workers=True,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            sintel_clean_train, sintel_final_train, kitti2015_train,
            kitti2012_train
        ]))
val_dataloader = [
    dict(
        batch_size=1,
        num_workers=2,
        sampler=dict(type='DefaultSampler', shuffle=False),
        drop_last=False,
        persistent_workers=True,
        dataset=sintel_clean_test),
    dict(
        batch_size=1,
        num_workers=2,
        sampler=dict(type='DefaultSampler', shuffle=False),
        drop_last=False,
        persistent_workers=True,
        dataset=sintel_final_test),
    dict(
        batch_size=1,
        num_workers=2,
        sampler=dict(type='DefaultSampler', shuffle=False),
        drop_last=False,
        persistent_workers=True,
        dataset=kitti2015_val_test)
]
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='EndPointError', prefix='clean'),
    dict(type='EndPointError', prefix='final'),
    [
        dict(type='EndPointError', prefix='KITTI2015'),
        dict(type='FlowOutliers', prefix='KITTI2015')
    ]
]
test_evaluator = val_evaluator
