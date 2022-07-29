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
    dict(type='PackFlowInputs')
]
sintel_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputPad', exponent=3),
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

sintel_clean_train_x100 = dict(
    type='RepeatDataset', times=100, dataset=sintel_clean_train)
sintel_final_train_x100 = dict(
    type='RepeatDataset', times=100, dataset=sintel_final_train)

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
    dict(type='PackFlowInputs')
]
flyingthings3d_clean_train = dict(
    type='FlyingThings3D',
    data_root='data/flyingthings3d',
    pipeline=flyingthing3d_train_pipeline,
    test_mode=False,
    pass_style='clean',
    scene='left',
    double=True)

train_dataloader = dict(
    batch_size=2,
    num_workers=5,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    drop_last=True,
    persistent_workers=True,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            sintel_clean_train_x100, sintel_final_train_x100,
            flyingthings3d_clean_train
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
        dataset=sintel_final_test)
]
test_dataloader = val_dataloader
val_evaluator = [
    dict(type='EndPointError', prefix='clean'),
    dict(type='EndPointError', prefix='final')
]
test_evaluator = val_evaluator
