img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False)

crop_size = (384, 768)

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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0., 1.)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=global_transform,
        relative_transform=relative_transform,
        check_bound=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'occ_gt'],
        meta_keys=[
            'img_fields', 'ann_fields', 'filename1', 'filename2',
            'ori_filename1', 'ori_filename2', 'filename_flow',
            'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputResize', exponent=6),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape'
        ])
]

sintel_clean_train = dict(
    type='Sintel',
    pipeline=train_pipeline,
    data_root='data/Sintel',
    test_mode=False,
    pass_style='clean')

sintel_final_train = dict(
    type='Sintel',
    pipeline=train_pipeline,
    data_root='data/Sintel',
    test_mode=False,
    pass_style='final')

sintel_clean_test = dict(
    type='Sintel',
    pipeline=test_pipeline,
    data_root='data/Sintel',
    test_mode=True,
    pass_style='clean')

sintel_final_test = dict(
    type='Sintel',
    pipeline=test_pipeline,
    data_root='data/Sintel',
    test_mode=True,
    pass_style='final')

data = dict(
    train_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=5,
        drop_last=True,
        persistent_workers=True),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=5,
        shuffle=False,
        persistent_workers=True),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    train=[sintel_clean_train, sintel_final_train],
    val=dict(
        type='ConcatDataset',
        datasets=[sintel_clean_test, sintel_final_test],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[sintel_clean_test, sintel_final_test],
        separate_eval=True))
