img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False)

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
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    test=kitti2015_val_test)
