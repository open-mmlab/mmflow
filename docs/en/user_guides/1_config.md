# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

## Config File Structure

There are 4 basic component types under `config/_base_`, datasets, models, schedules, default_runtime.
Many methods could be easily constructed with one of each like PWC-Net.
The configs that are composed by components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config.
All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from existing methods.
For example, if some modification is made based on PWC-Net, users may first inherit the basic PWC-Net structure by
specifying `_base_ = ../pwcnet/pwcnet_8xb1_slong_flyingchairs-384x448.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods,
you may create a folder `xxx` under `configs`.

Please refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) for detailed documentation.

## Config File Naming Convention

We follow the below style to name config files. Contributors are advised to follow the same style.

```text
{model}_[gpu x batch_per_gpu]_{schedule}_{training datasets}-[input_size].py
```

`{xxx}` is a required field and `[yyy]` is optional.

- `{model}`: model type like `pwcnet`, `flownets`, etc.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU, like `8xb1`.
- `{schedule}`: training schedule. Following FlowNet2's convention, we use `slong`, `sfine` and `sshort`, or number of iteration
  like `150k` 150k(iterations).
- `{training datasets}`: training dataset like `flyingchairs`, `flyingthings3d_subset`, `flyingthings3d`.
- `[input_size]`: the size of training images.

## Config System

To help the users have a basic idea of a complete config and the modules in MMFlow,
we make brief comments on the config of PWC-Net trained on FlyingChairs with slong schedule.
For more detailed usage and the corresponding alternative for each module,
please refer to the API documentation and
the [tutorial](https://github.com/open-mmlab/mmdetection/blob/rangilyu/3.x-config-doc/docs/en/tutorials/config.md) in MMDetection.

```python
_base_ = [
    '../_base_/models/pwcnet.py', '../_base_/datasets/flyingchairs_384x448.py',
    '../_base_/schedules/schedule_s_long.py', '../_base_/default_runtime.py'
]  # base config file which we build new config file on.
```

`_base_/models/pwcnet.py` is a basic model cfg file for PWC-Net.

```python
model = dict(
    type='PWCNet',  # The algorithm name.
    data_preprocessor=dict(  # The config of data preprocessor, usually includes image normalization and augmentation.
        type='FlowDataPreprocessor',  # The type of data preprocessor.
        mean=[0., 0., 0.],  # Mean values used for normalizing the input images.
        std=[255., 255., 255.],  # Standard variance used for normalizing the input images.
        bgr_to_rgb=False,  # Whether to convert image from BGR to RGB.
        sigma_range=(0, 0.04),  # Add gaussian noise for data augmentation, the sigma is uniformly sampled from [0, 0.04].
        clamp_range=(0., 1.)),  # After adding gaussian noise, clamp the range to [0., 1.].
    encoder=dict(  # Encoder module config
        type='PWCNetEncoder',  # The name of encoder in PWC-Net.
        in_channels=3,  # The input channels.
        # The type of this sub-module, if net_type is Basic, then the number of convolution layers of each level is 3,
        # if net_type is Small, the the number of convolution layers of each level is 2.
        net_type='Basic',
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],  # The list of feature pyramid levels that are the keys for output dict.
        out_channels=(16, 32, 64, 96, 128, 196),   # List of numbers of output channels of each pyramid level.
        strides=(2, 2, 2, 2, 2, 2),  # List of strides of each pyramid level.
        dilations=(1, 1, 1, 1, 1, 1),  # List of dilation of each pyramid level.
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),  # Config dict for each activation layer in ConvModule.
    decoder=dict(  # Decoder module config.
        type='PWCNetDecoder',  # The name of flow decoder in PWC-Net.
        in_channels=dict(
            level6=81, level5=213, level4=181, level3=149, level2=117),  # Input channels of basic dense block.
        flow_div=20.,  # The constant divisor to scale the ground truth value.
        corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
        warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled=False,  # Whether to use scaled correlation by the number of elements involved to calculate correlation or not.
        post_processor=dict(type='ContextNet', in_channels=565),  # The configuration for post processor.
        flow_loss=dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights={  # The weights for different levels of flow.
                'level2': 0.005,
                'level3': 0.01,
                'level4': 0.02,
                'level5': 0.08,
                'level6': 0.32
            }),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
    init_cfg=dict(
        type='Kaiming',
        nonlinearity='leaky_relu',
        layer=['Conv2d', 'ConvTranspose2d'],
        mode='fan_in',
        bias=0))
randomness = dict(seed=0, diff_rank_seed=True)  # Random seed.
```

in `_base_/datasets/flyingchairs_384x448.py`

```python
dataset_type = 'FlyingChairs'  # Dataset type, which will be used to define the dataset.
data_root = 'data/FlyingChairs_release'  # Root path of the dataset.

# global_transform and relative_transform are intermediate variables used in RandomAffine
# Keys of global_transform and relative_transform should be the subset of
#     ('translates', 'zoom', 'shear', 'rotate'). And also, each key and its
#     corresponding values has to satisfy the following rules:
#         - translates: the translation ratios along x axis and y axis. Defaults
#             to(0., 0.).
#         - zoom: the min and max zoom ratios. Defaults to (1.0, 1.0).
#         - shear: the min and max shear ratios. Defaults to (1.0, 1.0).
#         - rotate: the min and max rotate degree. Defaults to (0., 0.).
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

file_client_args = dict(backend='disk')  # File client arguments.

train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromFile', file_client_args=file_client_args),  # Load images.
    dict(type='LoadAnnotations', file_client_args=file_client_args),  # Load flow data.
    dict(
        type='ColorJitter',  # Randomly change the brightness, contrast, saturation and hue of an image.
        brightness=0.5,  # How much to jitter brightness.
        contrast=0.5,  # How much to jitter contrast.
        saturation=0.5,  # How much to jitter saturation.
        hue=0.5),  # How much to jitter hue.
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),  # Randomly gamma correction on images.
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),  # Random horizontal flip.
    dict(type='RandomFlip', prob=0.5, direction='vertical'),  # Random vertical flip.
    dict(
        type='RandomAffine',  # Random affine transformation of images.
        global_transform=global_transform,  # See comments above for global_transform.
        relative_transform=relative_transform),  # See comments above for relative_transform.
    dict(type='RandomCrop', crop_size=(384, 448)),  # Random crop the image and flow as (384, 448).
    dict(type='PackFlowInputs')  # Format the annotation data and decide which keys in the data should be packed into data_samples.
]

test_pipeline = [  # Testing pipeline.
    dict(type='LoadImageFromFile'),  # Load images.
    dict(type='LoadAnnotations'),  # Load flow data.
    dict(type='InputResize', exponent=6),  # Resize the width and height of the input images to a multiple of 2^6.
    dict(type='PackFlowInputs')  # Format the annotation data and decide which keys in the data should be packed into data_samples.
]

# flyingchairs_train and flyingchairs_test are intermediate variables used in dataloader,
# they define the type, pipeline, root path and split file of FlyingChairs.
flyingchairs_train = dict(
    type=dataset_type,
    pipeline=train_pipeline,
    data_root=data_root,
    split_file='data/FlyingChairs_release/FlyingChairs_train_val.txt')  # train-validation split file.
flyingchairs_test = dict(
    type=dataset_type,
    pipeline=test_pipeline,
    data_root=data_root,
    test_mode=True,  # Use test set.
    split_file='data/FlyingChairs_release/FlyingChairs_train_val.txt')  # train-validation split file

train_dataloader = dict(
    batch_size=1,  # Batch size of a single GPU.
    num_workers=2,  # Worker to pre-fetch data for each single GPU.
    sampler=dict(type='InfiniteSampler', shuffle=True),  # Randomly shuffle during training.
    drop_last=True,  # Drop the last non-full batch during training.
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate training speed.
    dataset=flyingchairs_train)

val_dataloader = dict(
    batch_size=1,  # Batch size of a single GPU.
    num_workers=2,  # Worker to pre-fetch data for each single GPU.
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing.
    drop_last=False,  # No need to drop the last non-full batch.
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate training speed.
    dataset=flyingchairs_test)
test_dataloader = val_dataloader

# The metric to measure the accuracy. Here, we use EnePointError.
val_evaluator = dict(type='EndPointError')
test_evaluator = val_evaluator
```

in `_base_/schedules/schedule_s_long.py`

```python
# training schedule for S_long schedule
train_cfg = dict(by_epoch=False, max_iters=1200000, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001, weight_decay=0.0004, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    gamma=0.5,
    milestones=[400000, 600000, 800000, 1000000])

# default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Log the time spent during iteration.
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),  # Collect and write logs from different components of ``Runner``.
    param_scheduler=dict(type='ParamSchedulerHook'),  # update some hyper-parameters in optimizer, e.g., learning rate.
    checkpoint=dict(type='CheckpointHook', interval=100000, by_epoch=False),  # Save checkpoints periodically.
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Data-loading sampler for distributed training.
    visualization=dict(type='FlowVisualizationHook'))  # Show or Write the predicted results during the process of testing and validation.
```

in `_base_/default_runtime.py`

```python
# Set the default scope of the registry to mmflow.
default_scope = 'mmflow'

# environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualizer
vis_backends = [dict(type='LocalVisBackend')]  # The backend of visualizer.
visualizer = dict(
    type='FlowLocalVisualizer', vis_backends=vis_backends, name='visualizer')

resume = False  # Whether to resume from existed model.
```

## Modify config through script arguments

When submitting jobs using "tools/train.py" or "tools/test.py", you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-option model.encoder.in_channels=6`.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromWebcam'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.0.type=LoadImageFromWebcam`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `sigma_range=(0, 0.04)` in `data_preprocessor` of `model`.
  If you want to change this key, you may specify in two ways:

  1. `--cfg-options model.data_preprocessor.sigma_range="(0, 0.05)"`. Note that the quotation mark " is necessary to support list/tuple data types.
  2. `--cfg-options model.data_preprocessor.sigma_range=0,0.05`. Note that **NO** white space is allowed in the specified value.
     In addition, if the original type is tuple, it will be automatically converted to list after this way.

```{note}
This modification of only supports modifying configuration items of string, int, float, boolean, None, list and tuple types.
More specifically, for list and tuple types, the elements inside them must also be one of the above seven types.
```

## FAQ

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some fields in base configs.
You may refer to mmengine for simple illustration.

You may have a careful look at [this tutorial](https://github.com/open-mmlab/mmdetection/blob/rangilyu/3.x-config-doc/docs/en/tutorials/config.md#ignore-some-fields-in-the-base-configs) for better understanding of this feature.

### Use intermediate variables in configs

Some intermediate variables are used in the config files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, users need to pass the intermediate variables into corresponding fields again.
For example, the original `pwcnet_8xb1_slong_flyingchairs-384x448.py` is

```python
_base_ = [
    '../_base_/models/pwcnet.py', '../_base_/datasets/flyingchairs_384x448.py',
    '../_base_/schedules/schedule_s_long.py', '../_base_/default_runtime.py'
]
```

According to the setting: `vis_backends = [dict(type='LocalVisBackend')]` in `_base_/default_runtime.py`, we can only store the visualization results locally.
If we want to store them on Tensorboard as well, then the `vis_backends` is the intermediate variable we would like to modify.

```python
_base_ = [
    '../_base_/models/pwcnet.py', '../_base_/datasets/flyingchairs_384x448.py',
    '../_base_/schedules/schedule_s_long.py', '../_base_/default_runtime.py'
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='FlowLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

We first define the new `vis_backends` and pass them into `visualizer` again.
