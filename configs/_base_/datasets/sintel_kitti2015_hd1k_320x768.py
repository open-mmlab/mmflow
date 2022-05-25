img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False)

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
        global_transform=sintel_global_transform,
        relative_transform=sintel_relative_transform),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt'],
        meta_keys=[
            'img_fields', 'ann_fields', 'filename1', 'filename2',
            'ori_filename1', 'ori_filename2', 'filename_flow',
            'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg'
        ]),
]

sintel_test_pipeline = [
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
    test_mode=True,
    pass_style='clean')

sintel_final_test = dict(
    type='Sintel',
    pipeline=sintel_test_pipeline,
    data_root='data/Sintel',
    test_mode=True,
    pass_style='final')

kitti_global_transform = dict(
    translates=(0.02, 0.02),
    zoom=(0.98, 1.02),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))

kitti_relative_transform = dict(
    translates=(0.0025, 0.0025),
    zoom=(0.99, 1.01),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))

kitti_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(
        type='ColorJitter',
        brightness=0.05,
        contrast=0.2,
        saturation=0.25,
        hue=0.1),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GaussianNoise', sigma_range=(0, 0.02), clamp_range=(0., 1.)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=kitti_global_transform,
        relative_transform=kitti_relative_transform),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'img_fields', 'ann_fields', 'filename1', 'filename2',
            'ori_filename1', 'ori_filename2', 'filename_flow',
            'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg'
        ]),
]

kitti2015_train = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=kitti_train_pipeline,
    test_mode=False)

hd1k_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(type='RandomCrop', crop_size=(436, 1024)),
    dict(
        type='ColorJitter',
        brightness=0.05,
        contrast=0.2,
        saturation=0.25,
        hue=0.1),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='GaussianNoise', sigma_range=(0, 0.02), clamp_range=(0., 1.)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=kitti_global_transform,
        relative_transform=kitti_relative_transform),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'img_fields', 'ann_fields', 'filename1', 'filename2',
            'ori_filename1', 'ori_filename2', 'filename_flow',
            'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg'
        ]),
]

hd1k_train = dict(
    type='HD1K',
    data_root='data/hd1k',
    pipeline=hd1k_train_pipeline,
    test_mode=False),

data = dict(
    train_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        drop_last=True,
        sample_ratio=(0.5, 0.25, 0.25),
        persistent_workers=True),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=5,
        shuffle=False,
        persistent_workers=True),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    train=[[sintel_clean_train, sintel_final_train], kitti2015_train,
           hd1k_train],
    val=dict(
        type='ConcatDataset',
        datasets=[sintel_clean_test, sintel_final_test],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[sintel_clean_test, sintel_final_test],
        separate_eval=True))
