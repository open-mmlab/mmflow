# Data Transforms

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

Here is a pipeline example for PWC-Net training on FlyingChairs.

```python
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', file_client_args=file_client_args),
    dict(
        type='ColorJitter',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
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
    dict(type='PackFlowInputs')
]
```

For each operation, we list the related dict fields that are added/updated/removed.
Before pipelines, the information we can directly obtain from the datasets are img1_path, img2_path and flow_fw_path.

### Data loading

`LoadImageFromFile`

- add: img1, img2, img_shape, ori_shape

`LoadAnnotations`

- add: gt_flow_fw, gt_flow_bw(None), sparse(False)

```{note}
FlyingChairs doesn't provide the ground truth of backward flow, so gt_flow_bw is None.
Besides, FlyingChairs' ground truth is dense, so sparse is False.
For some special datasets, such as HD1K and KITTI, their ground truth is sparse, so gt_valid_fw and gt_valid_bw will be added.
FlyingChairsOcc and FlyingThing3d contain the ground truth of occlusion, so gt_occ_fw and gt_occ_bw will be added for these datasets.
In the pipelines below, we only consider the case of FlyingChairs.
```

### Pre-processing

`ColorJitter`

- update: img1, img2

`RandomGamma`

- add: gamma
- update: img1, img2

`RandomFlip`

- add: flip, flip_direction
- update: img1, img2, flow_gt

`RandomAffine`

- add: global_ndc_affine_mat, relative_ndc_affine_mat
- update: img1, img2, flow_gt

`RandomCrop`

- add: crop_bbox
- update: img1, img2, flow_gt, img_shape

### Formatting

`PackFlowInputs`

- add: inputs, data_sample
- remove: img1 and img2 (merged into inputs), keys specified by `data_keys` (like gt_flow_fw, merged into data_sample)
  keys specified by `meta_keys` (merged into the metainfo of data_sample), all other keys
