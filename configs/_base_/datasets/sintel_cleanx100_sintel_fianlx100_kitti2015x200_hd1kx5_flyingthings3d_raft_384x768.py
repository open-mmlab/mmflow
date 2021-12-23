img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
crop_size = (368, 768)

# Sintel config
sintel_train_pipeline = [
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
        crop_size=crop_size,
        min_scale=-0.2,
        max_scale=0.6,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.1, direction='vertical'),
    dict(type='Validation', max_flow=1000.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
            'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
            'erase_bounds', 'erase_num', 'scale_factor'
        ])
]
sintel_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputPad', exponent=3),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape', 'pad'
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

sintel_clean_train_x100 = dict(
    type='RepeatDataset', times=100, dataset=sintel_clean_train)
sintel_final_train_x100 = dict(
    type='RepeatDataset', times=100, dataset=sintel_final_train)

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

# KITTI config
kitti_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(
        type='ColorJitter',
        asymmetric_prob=0.0,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.5 / 3.14),
    dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
    dict(
        type='SpacialTransform',
        spacial_prob=0.8,
        stretch_prob=0.8,
        crop_size=crop_size,
        min_scale=-0.3,
        max_scale=0.5,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.1, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
            'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
            'erase_bounds', 'erase_num', 'scale_factor'
        ])
]
kitti_train = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=kitti_train_pipeline,
    test_mode=False)
kitti_train_x200 = dict(type='RepeatDataset', times=200, dataset=kitti_train)

# Flyingthings3d config
flyingthing3d_train_pipeline = [
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
        crop_size=crop_size,
        min_scale=-0.4,
        max_scale=0.8,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.1, direction='vertical'),
    dict(type='Validation', max_flow=1000.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
            'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
            'erase_bounds', 'erase_num', 'scale_factor'
        ])
]
flyingthings3d_clean_train = dict(
    type='FlyingThings3D',
    data_root='data/flyingthings3d',
    pipeline=flyingthing3d_train_pipeline,
    pass_style='clean',
    scene='left')

# HD1K config
hd1k_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(
        type='ColorJitter',
        asymmetric_prob=0.,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.5 / 3.14),
    dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
    dict(
        type='SpacialTransform',
        spacial_prob=0.8,
        stretch_prob=0.8,
        crop_size=crop_size,
        min_scale=-0.5,
        max_scale=0.2,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.1, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
            'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
            'erase_bounds', 'erase_num', 'scale_factor'
        ])
]
hd1k_train = dict(
    type='HD1K',
    pipeline=hd1k_train_pipeline,
    data_root='data/hd1k',
    test_mode=False)

hd1k_train_x5 = dict(type='RepeatDataset', times=5, dataset=hd1k_train)

data = dict(
    train_dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=5,
        drop_last=True,
        shuffle=True,
        persistent_workers=True),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=5,
        shuffle=False,
        persistent_workers=True),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    train=[
        sintel_clean_train_x100, sintel_final_train_x100, kitti_train_x200,
        flyingthings3d_clean_train, hd1k_train_x5
    ],
    val=dict(
        type='ConcatDataset',
        datasets=[sintel_clean_test, sintel_final_test],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[sintel_clean_test, sintel_final_test],
        separate_eval=True))
