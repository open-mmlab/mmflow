# Tutorial 3: Custom Data Pipelines

## Design of Data pipelines

Following typical conventions, we use `Dataset` and `DataLoader` for data loading
with multiple workers. `Dataset` returns a dict of data items corresponding
the arguments of models' forward method.
Since the data flow estimation may not be the same size, we introduce a new `DataContainer` type in MMCV to help collect and distribute
data of different size.
See [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py) for more details.

The data preparation pipeline and the dataset is decomposed. Usually a dataset
defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.
A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next transform.

The operations are categorized into data loading, pre-processing, formatting.

Here is a pipeline example for PWC-Net

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5,
         hue=0.5),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(type='Normalize', mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False),
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0., 1.)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomAffine',
         global_transform=dict(
            translates=(0.05, 0.05),
            zoom=(1.0, 1.5),
            shear=(0.86, 1.16),
            rotate=(-10., 10.)
        ),
         relative_transform=)dict(
            translates=(0.00375, 0.00375),
            zoom=(0.985, 1.015),
            shear=(1.0, 1.0),
            rotate=(-1.0, 1.0)
        ),
    dict(type='RandomCrop', crop_size=(384, 448)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt'],
        meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg']),
]

```

For each operation, we list the related dict fields that are added/updated/removed.

### Data loading

`LoadImageFromFile`

- add: img1, img2, filename1, filename2, img_shape, ori_shape, pad_shape, scale_factor, img_norm_cfg

`LoadAnnotations`

- add: flow_gt, filename_flow

### Pre-processing

`ColorJitter`

- update: img1, img2

`RandomGamma`

- update: img1, img2

`Normalize`

- update: img1, img2, img_norm_cfg

`GaussianNoise`

- update: img1, img2

`RandomFlip`

- update: img1, img2, flow_gt

`RandomAffine`

- update: img1, img2, flow_gt

`RandomCrop`

- update: img1, img2, flow_gt, img_shape

### Formatting

`DefaultFormatBundle`

- update: img1, img2, flow_gt

`Collect`

- add: img_meta (the keys of img_meta is specified by `meta_keys`)
- remove: all other keys except for those specified by `keys`

## Extend and use custom pipelines

1. Write a new pipeline in any file, e.g., `my_pipeline.py`. It takes a dict as input and return a dict.

   ```python
   from mmflow.datasets import PIPELINES

   @PIPELINES.register_module()
   class MyTransform:

       def __call__(self, results):
           results['dummy'] = True
           return results
   ```

2. Import the new class.

   ```python
   from .my_pipeline import MyTransform
   ```

3. Use it in config files.

   ```python
   train_pipeline = [
   dict(type='LoadImageFromFile'),
   dict(type='LoadAnnotations'),
   dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5,
        hue=0.5),
   dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
   dict(type='Normalize', mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=False),
   dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0., 1.)),
   dict(type='RandomFlip', prob=0.5, direction='horizontal'),
   dict(type='RandomFlip', prob=0.5, direction='vertical'),
   dict(type='RandomAffine',
        global_transform=dict(
           translates=(0.05, 0.05),
           zoom=(1.0, 1.5),
           shear=(0.86, 1.16),
           rotate=(-10., 10.)
       ),
        relative_transform=)dict(
           translates=(0.00375, 0.00375),
           zoom=(0.985, 1.015),
           shear=(1.0, 1.0),
           rotate=(-1.0, 1.0)
       ),
   dict(type='RandomCrop', crop_size=(384, 448)),
   dict(type='MyTransform'),
   dict(type='DefaultFormatBundle'),
   dict(
       type='Collect',
       keys=['imgs', 'flow_gt'],
       meta_keys=('img_fields', 'ann_fields', 'filename1', 'filename2',
                  'ori_filename1', 'ori_filename2', 'filename_flow',
                  'ori_filename_flow', 'ori_shape', 'img_shape',
                  'img_norm_cfg'))]
   ```
