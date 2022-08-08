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
    dict(type='PackFlowInputs')
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
    dict(type='PackFlowInputs')
]

kitti2015_val_test = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=kitti_test_pipeline,
    test_mode=False)

train_dataloader = dict(
    batch_size=2,
    num_workers=5,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    drop_last=True,
    persistent_workers=True,
    dataset=kitti_train)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=False,
    persistent_workers=True,
    dataset=kitti2015_val_test)
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='EndPointError', prefix='KITTI2015'),
    dict(type='FlowOutliers', prefix='KITTI2015')
]
test_evaluator = val_evaluator
