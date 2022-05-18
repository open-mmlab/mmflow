# Tutorial 0: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

## Config File Structure

There are 4 basic component types under `config/_base_`, datasets, models, schedules, default_runtime.
Many methods could be easily constructed with one of each like PWC-Net.
The configs that are composed by components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config.
All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from existing methods.
For example, if some modification is made base on PWC-Net, user may first inherit the basic PWC-Net structure by
specifying `_base_ = ../pwcnet/pwcnet_slong_8x1_flyingchairs_384x448.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods,
you may create a folder `xxx` under `configs`.

Please refer to [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) for detailed documentation.

## Config File Naming Convention

We follow the below style to name config files. Contributors are advised to follow the same style.

```text
{model}_{schedule}_[gpu x batch_per_gpu]_{training datasets}_[input_size].py
```

`{xxx}` is a required field and `[yyy]` is optional.

- `{model}`: model type like `pwcnet`, `flownets`, etc.
- `{schedule}`: training schedule. Following FlowNet2's convention, we use `slong`, `sfine` and `sshort`, or number of iteration
  like `150k` 150k(iterations).
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU, like `8x1`.
- `{training datasets}`: training dataset like `flyingchairs`, `flyingthings3d_subset`, `flyingthings3d`.
- `[input_size]`: the size of training images.

## Config System

To help the users have a basic idea of a complete config and the modules in MMFlow,
we make brief comments on the config of PWC-Net trained on FlyingChairs with slong schedule.
For more detailed usage and the corresponding alternative for each module,
please refer to the API documentation and
the [tutorial in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md).

```python
_base_ = [
    '../_base_/models/pwcnet.py', '../_base_/datasets/flyingchairs_384x448.py',
    '../_base_/schedules/schedule_s_long.py', '../_base_/default_runtime.py'
]# base config file which we build new config file on.
```

`_base_/models/pwc_net.py` is a basic model cfg file for PWC-Net.

```python
model = dict(
    type='PWCNet',  # The algorithm name
    encoder=dict(  # Encoder module config
        type='PWCNetEncoder',  # The name of encoder in PWC-Net.
        in_channels=3,  # The input channels
        #  The type of this sub-module, if net_type is Basic, the the number of convolution layers of each level is 3,
        #  if net_type is Small, the the number of convolution layers of each level is 2.
        net_type='Basic',
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ], # The list of feature pyramid levels that are the keys for output dict.
        out_channels=(16, 32, 64, 96, 128, 196),  #  List of numbers of output channels of each pyramid level.
        strides=(2, 2, 2, 2, 2, 2),  # List of strides of each pyramid level.
        dilations=(1, 1, 1, 1, 1, 1),  # List of dilation of each pyramid level.
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),  # Config dict for each activation layer in ConvModule.
    decoder=dict(  # Decoder module config.
        type='PWCNetDecoder',  # The name of flow decoder in PWC-Net.
        in_channels=dict(
            level6=81, level5=213, level4=181, level3=149, level2=117),  # Input channels of basic dense block.
        flow_div=20.,  # The constant divisor to scale the ground truth value.
        corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
        warp_cfg=dict(type='Warp'),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled=False,  # Whether to use scaled correlation by the number of elements involved to calculate correlation or not.
        post_processor=dict(type='ContextNet', in_channels=565),  # The configuration for post processor.
        flow_loss=dict(  # The loss function configuration.
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights={ # The weights for different levels of flow.
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
```

in `_base_/datasets/flyingchairs_384x448.py`

```python
dataset_type = 'FlyingChairs'  # Dataset name
data_root = 'data/FlyingChairs/data'  # Root path of dataset

img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255], to_rgb=False)  # Image normalization config to normalize the input images

train_pipeline = [ # Training pipeline
    dict(type='LoadImageFromFile'),  # load images
    dict(type='LoadAnnotations'),  # load flow data
    dict(type='ColorJitter',  # Randomly change the brightness, contrast, saturation and hue of an image.
     brightness=0.5,  # How much to jitter brightness.
     contrast=0.5,  # How much to jitter contrast.
     saturation=0.5,  # How much to jitter saturation.
         hue=0.5),  # How much to jitter hue.
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),  # Randomly gamma correction on images.
    dict(type='Normalize', **img_norm_cfg),  # Normalization config, the values are from img_norm_cfg
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0., 1.)),  # Add Gaussian noise and a sigma uniformly sampled from [0, 0.04];
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),  # Random horizontal flip
    dict(type='RandomFlip', prob=0.5, direction='vertical'),   # Random vertical flip
    # Random affine transformation of images
    # Keys of global_transform and relative_transform should be the subset of
    #     ('translates', 'zoom', 'shear', 'rotate'). And also, each key and its
    #     corresponding values has to satisfy the following rules:
    #         - translates: the translation ratios along x axis and y axis. Defaults
    #             to(0., 0.).
    #         - zoom: the min and max zoom ratios. Defaults to (1.0, 1.0).
    #         - shear: the min and max shear ratios. Defaults to (1.0, 1.0).
    #         - rotate: the min and max rotate degree. Defaults to (0., 0.).
    dict(type='RandomAffine',
         global_transform=dict(
            translates=(0.05, 0.05),
            zoom=(1.0, 1.5),
            shear=(0.86, 1.16),
            rotate=(-10., 10.)
        ),
         relative_transform=dict(
            translates=(0.00375, 0.00375),
            zoom=(0.985, 1.015),
            shear=(1.0, 1.0),
            rotate=(-1.0, 1.0)
        )),
    dict(type='RandomCrop', crop_size=(384, 448)),  # Random crop the image and flow as (384, 448)
    dict(type='DefaultFormatBundle'),  # It simplifies the pipeline of formatting common fields, including "img1", "img2" and "flow_gt".
    dict(
        type='Collect',  # Collect data from the loader relevant to the specific task.
        keys=['imgs', 'flow_gt'],
        meta_keys=('img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputResize', exponent=4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='TestFormatBundle'),  # It simplifies the pipeline of formatting common fields, including "img1"
    # and "img2".
    dict(
        type='Collect',
        keys=['imgs'],  # Collect data from the loader relevant to the specific task.
        meta_keys=('flow_gt', 'filename1', 'filename2', 'ori_filename1',
                   'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'scale_factor', 'pad_shape'))  # 'flow_gt' in img_meta is works for online evaluation.
]

data = dict(
    train_dataloader=dict(
        samples_per_gpu=1,  # Batch size of a single GPU
        workers_per_gpu=5,  # Worker to pre-fetch data for each single GPU
        drop_last=True),  # Drops the last non-full batch

    val_dataloader=dict(
        samples_per_gpu=1,  # Batch size of a single GPU
        workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
        shuffle=False),  # Whether shuffle dataset.

    test_dataloader=dict(
        samples_per_gpu=1,  # Batch size of a single GPU
        workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
        shuffle=False),  # Whether shuffle dataset.

    train=dict(  # Train dataset config
        type=dataset_type,
        pipeline=train_pipeline,
        data_root=data_root,
        split_file='data/FlyingChairs_release/FlyingChairs_train_val.txt',  # train-validation split file
    ),

    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        test_mode=True),

    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        test_mode=True)
)
```

in `_base_/schedules/schedule_s_long.py`

```python
# optimizer
optimizer = dict(
    type='Adam', lr=0.0001, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[400000, 600000, 800000, 1000000])
runner = dict(type='IterBasedRunner', max_iters=1200000)
checkpoint_config = dict(by_epoch=False, interval=100000)
evaluation = dict(interval=100000, metric='EPE')
```

in `_base_/default_runtime.py`

```python
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])  # The logger used to record the training process.
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The level of logging.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
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

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark " is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## FAQ

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of fields in base configs.
You may refer to [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#inherit-from-base-config-with-ignored-fields) for simple illustration.

You may have a careful look at [this tutorial](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md) for better understanding of this feature.

### Use intermediate variables in configs

Some intermediate variables are used in the config files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, users need to pass the intermediate variables into corresponding fields again. An intuitive example can be found in [this tutorial](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md).
