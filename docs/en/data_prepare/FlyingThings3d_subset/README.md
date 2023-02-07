# Prepare FlyingThing3d_subset dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{MIFDB16,
  author    = "N. Mayer and E. Ilg and P. H{\"a}usser and P. Fischer and D. Cremers and A. Dosovitskiy and T. Brox",
  title     = "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation",
  booktitle = "IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)",
  year      = "2016",
  note      = "arXiv:1512.02134",
  url       = "http://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16"
}
```

## Download and Unpack dataset

You can download datasets via \[BitTorrent\] (https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_image_clean.tar.bz2.torrent). Then, you need to unzip and move corresponding datasets to follow the folder structure shown above. The datasets have been well-prepared by the original authors.

```text
├── FlyingThings3D_subset
|   ├── train
|   |   ├── flow
|   |   |   ├── left
|   |   |   |    ├── into_future
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |    ├── into_past
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   ├── right
|   |   |   |    ├── into_future
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |    ├── into_past
|   |   |   |    |      ├── xxxxxxx.flo
|   |   ├── flow_occlusions
|   |   |   ├── left
|   |   |   |    ├── into_future
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |    ├── into_past
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   ├── right
|   |   |   |    ├── into_future
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |    ├── into_past
|   |   |   |    |      ├── xxxxxxx.flo
|   |   ├── image_clean
|   |   |   ├── left
|   |   |   |    ├── xxxxxxx.png
|   |   |   ├── right
|   |   |   |    ├── xxxxxxx.png
|   ├── val
|   |   ├── flow
|   |   |   ├── left
|   |   |   |    ├── into_future
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |    ├── into_past
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   ├── right
|   |   |   |    ├── into_future
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |    ├── into_past
|   |   |   |    |      ├── xxxxxxx.flo
|   |   ├── flow_occlusions
|   |   |   ├── left
|   |   |   |    ├── into_future
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |    ├── into_past
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   ├── right
|   |   |   |    ├── into_future
|   |   |   |    |      ├── xxxxxxx.flo
|   |   |   |    ├── into_past
|   |   |   |    |      ├── xxxxxxx.flo
|   |   ├── image_clean
|   |   |   ├── left
|   |   |   |    ├── xxxxxxx.png
|   |   |   ├── right
|   |   |   |    ├── xxxxxxx.png
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.

```bash
python tools/dataset_converters/prepare_flyingthings3d_subset.py [optional arguments]
```

This script accepts these arguments:

- `--data-root ${DATASET_DIR}`: The dataset directory of FlyingThings3D_subset, default to `'data/FlyingThings3D_subset'`.

- `--save-dir ${SAVE_DIR}`: The directory for saving the annotation file, default to`'data/FlyingThings3D_subset/'`,
  and annotation files for train and test dataset will be save as `${SAVE_DIR}/train.json` and `${SAVE_DIR}/test.json`

**Note**:

Annotation file is not required for local file storage, and it will be used in dataset config file when using cloud object storage like s3 storage. There is an example for using object storage:

```python
backend_args= dict(
    backend='s3',
    path_mapping=dict(
        {'data/': 's3://dataset_path'}))
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', backend_args=backend_args),
]
flyingthings3d_subset_train = dict(
    type='FlyingThings3DSubset',
    ann_file='train.json', # train.json is in data_root i.e. data/FlyingThings3D_subset/
    pipeline=train_pipeline,
    data_root='data/FlyingThings3D_subset',
    test_mode=False,
    scene='left')
```
