train_dataset_type = 'FlyingThings3D'
train_data_root = 'data/flyingthings3d'
test_dataset_type = 'Sintel'
test_data_root = 'data/Sintel'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
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
test_pipeline = [
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

train_dataset_cleanpass = dict(
    type=train_dataset_type,
    data_root=train_data_root,
    pipeline=train_pipeline,
    test_mode=False,
    pass_style='clean',
    scene='left')
train_dataset_finalpass = dict(
    type=train_dataset_type,
    data_root=train_data_root,
    pipeline=train_pipeline,
    test_mode=False,
    pass_style='final',
    scene='left')
test_data_cleanpass = dict(
    type=test_dataset_type,
    data_root=test_data_root,
    pipeline=test_pipeline,
    test_mode=True,
    pass_style='clean')
test_data_finalpass = dict(
    type=test_dataset_type,
    data_root=test_data_root,
    pipeline=test_pipeline,
    test_mode=True,
    pass_style='final')

data = dict(
    train_dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        drop_last=True,
        persistent_workers=True),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        shuffle=False,
        persistent_workers=True),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    train=[train_dataset_cleanpass, train_dataset_finalpass],
    val=dict(
        type='ConcatDataset',
        datasets=[test_data_cleanpass, test_data_finalpass],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[test_data_cleanpass, test_data_finalpass],
        separate_eval=True))
