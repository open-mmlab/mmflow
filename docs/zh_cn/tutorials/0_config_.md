# 教程0: 如何使用 Configs

我们的配置文件 (configs) 中支持了模块化和继承设计，这便于进行各种实验。如果需要检查配置文件，可以通过运行 `python tools/misc/print_config.py /PATH/TO/CONFIG` 来查看完整的配置文件。

## 配置文件结构

在目录 `config/_base_` 下有四种基本模块类型，即数据集 (datasets) 、模型 (models) 、训练计划 (schedules) 以及默认运行配置 (default_runtime)。很多模型可以很容易地参考其中一种方法 (如 PWC-Net )来构建。这些由 `_base_` 中的模块构成的配置文件被称为 *原始配置 (primitive configs)*。

对于同一文件夹下的所有配置，建议只有**一个**原始配置。所有其他配置都应该从原始配置继承。这样，最大继承级别 (inheritance level) 为 3。

简单来说，我们建议贡献者去继承现有模型的配置文件。例如，如果在 PWC-Net 的基础上做了一些改动，可以首先通过在配置文件中指定原始配置 `_base_ = ../pwcnet/pwcnet_slong_8x1_flyingchairs_384x448.py` 来继承基本的 PWC-Net 结构，然后再根据需要修改配置文件中的指定字段 (fields)。

如果需要搭建不能与现有模型共享任何结构的全新的模型，您可以在 `configs` 下创建一个新文件夹 `xxx`。

您也可以参考 [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) 文档中的更多细节。

## 配置文件命名规则

我们按照下面的风格来命名配置文件。建议贡献者使用相同的风格。

```text
{model}_{schedule}_[gpu x batch_per_gpu]_{training datasets}_[input_size].py
```

`{xxx}` 表示必填字段，而 `[yyy]` 表示可选字段。

- `{model}`: 模型类型，如 `pwcnet`, `flownets` 等等。
- `{schedule}`: 训练计划。按照 FlowNet2 中的约定，我们使用 `slong`、 `sfine` 和 `sshort`，或者像 `150k` 表示150k(iterations) 这样指定迭代次数。
- `[gpu x batch_per_gpu]`: GPU 数量以及每个 GPU上分配的样本数， 如 `8x1`。
- `{training datasets}`: 训练数据集，如 `flyingchairs`， `flyingthings3d_subset` 或 `flyingthings3d`。
- `[input_size]`: 训练时图片大小。

## 配置文件结构

为了帮助用户对完整的配置和 MMFlow 中的模块有一个基本的了解，我们以在 `flyingchairs` 上使用 `slong` 训练的 PWC-Net 的配置为例进行简单的讲解。有关每个模块的更详细的用法和相应的替代方案，请参阅 API 文档和 [MMDetection 教程](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md)。

```python
_base_ = [
    '../_base_/models/pwcnet.py', '../_base_/datasets/flyingchairs_384x448.py',
    '../_base_/schedules/schedule_s_long.py', '../_base_/default_runtime.py'
]# base config file which we build new config file on.
```

`_base_/models/pwc_net.py` 是 PWC-Net 模型的基本配置文件。

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

原始配置文件 `_base_/datasets/flyingchairs_384x448.py` 中是:

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

原始配置文件 `_base_/schedules/schedule_s_long.py` 中是:

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

原始配置文件 `_base_/default_runtime.py` 中是:

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

## 通过脚本参数来修改配置

在使用 `tools/train.py` 或 `tools/test.py` 时，可以通过指定 `--cfg-options` 来就地 (in-place) 修改配置。

- 更新配置字典链 (dict chains) 中的键 (keys)。

  可以按照原始配置中字典键的顺序指定配置选项。
  例如， `--cfg-option model.encoder.in_channels=6`。

- 更新配置列表中的键 (keys)。

  一些配置字典在配置文件中组成一个列表。例如，训练数据处理管道 (pipeline) `data.train.pipeline` 往往是一个像 `[dict(type='LoadImageFromFile'), ...]` 一样的列表。 如果希望将其中的 `'LoadImageFromFile'` 替换为 `'LoadImageFromWebcam'`，可以通过指定 `--cfg-option data.train.pipeline.0.type='LoadImageFromWebcam'` 来实现。

- 更新列表或元组的值。

  如果需要更新的值是一个元组或是列表。例如，配置文件中通常设置训练工作流为 `workflow=[('train', 1)]`。如果希望修改这个键值，可以通过指定 `--cfg-options workflow="[(train,1),(val,1)]"` 来实现。 注意，引号 \" 对于列表、元组数据类型是必需的，且在指定值的引号内**不允许**有空格。

## 常见问题 (FAQ)

### 忽略原始配置中的部分字段

如果需要，你可以通过设置 `_delete_=True` 来忽略原始配置文件中的部分字段。
可以参考 [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#inherit-from-base-config-with-ignored-fields) 中的简单说明。

请仔细阅读 [config 教程](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md) 以更好地了解这一方法。

### 使用配置文件中的中间变量

配置文件中使用了一些中间变量，例如数据集配置中的 `train_pipeline`/`test_pipeline`。
值得注意的是，在修改子配置文件中的中间变量时，用户需要再次将中间变量传递到相应的字段中。更为直观的例子参见 [config 教程](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md)。
