# Prepare KITTI flow2015 dataset

<!-- [DATASET] -->

```bibtex
@ARTICLE{Menze2018JPRS,
  author = {Moritz Menze and Christian Heipke and Andreas Geiger},
  title = {Object Scene Flow},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing (JPRS)},
  year = {2018}
}

@INPROCEEDINGS{Menze2015ISA,
  author = {Moritz Menze and Christian Heipke and Andreas Geiger},
  title = {Joint 3D Estimation of Vehicles and Scene Flow},
  booktitle = {ISPRS Workshop on Image Sequence Analysis (ISA)},
  year = {2015}
}
```

## Download and Unpack dataset

You can download datasets on this [webpage](http://www.cvlibs.net/datasets/kitti/user_login.php). Then, you need to unzip and move corresponding datasets to follow the folder structure shown below. The datasets have been well-prepared by the original authors.

```text
├── kitti2015
|   ├── training
|   |   ├── flow_occ
|   |   |   ├── xxxxxx_xx.png
|   |   ├── flow_noc
|   |   |   ├── xxxxxx_xx.png
|   |   ├── image_2
|   |   |   ├── xxxxxx_xx.png
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.

```bash
python tools/dataset_converters/prepare_kitti2015.py [optional arguments]
```

This script accepts these arguments:

- `--data-root ${DATASET_DIR}`: The dataset directory of FlyingChairs, default to `'data/kitti2015'`.

- `--save-dir ${SAVE_DIR}`: The directory for saving the annotation file, default to`'data/kitti2015/'`,
  and annotation files for train and test dataset will be save as `${SAVE_DIR}/train.json`.

**Note**:

Annotation file is not required for local file storage, and it will be used in dataset config file when using cloud object storage like s3 storage. There is an example for using object storage:

```python
backend_args= dict(
    backend='s3',
    path_mapping=dict(
        {'data/': 's3://dataset_path'}))
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', backend_args=backend_args, sparse=True)]
kitti_train = dict(
    type='KITTI2015',
    ann_file='train.json', # train.json is in data_root i.e. data/kitti2015/
    data_root='data/kitti2015',
    pipeline=kitti_train_pipeline,
    test_mode=False)
```
