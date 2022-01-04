img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)

crop_size = (288, 960)

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
        min_scale=-0.2,
        max_scale=0.4,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=crop_size),
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

kitti_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(type='InputPad', exponent=3),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'valid', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape', 'pad'
        ])
]

kitti2015_val_test = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=kitti_test_pipeline,
    test_mode=True)

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
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    train=kitti_train,
    val=kitti2015_val_test,
    test=kitti2015_val_test)
