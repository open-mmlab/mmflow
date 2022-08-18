# Prepare KITTI flow2012 dataset

<!-- [DATASET] -->

```bibtex
@INPROCEEDINGS{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}
```

## Download and Unpack dataset

You can download datasets on this [webpage](http://www.cvlibs.net/datasets/kitti/user_login.php). Then, you need to unzip and move corresponding datasets to follow the folder structure shown below. The datasets have been well-prepared by the original authors.

```text
kitti2012
|   |   ├── training
|   |   |   ├── flow_occ
|   |   |   |   ├── xxxxxx_xx.png
|   |   |   ├── flow_noc
|   |   |   |   ├── xxxxxx_xx.png
|   |   |   ├── colored_0
|   |   |   |   ├── xxxxxx_xx.png
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.

```bash
python tools/dataset_converters/prepare_kitti2012.py [optional arguments]
```

This scrip accepts these arguments:

- `--data-root ${DATASET_DIR}`: The dataset directory of FlyingChairs, default to `'data/kitti2012'`.

- `--save-dir ${SAVE_DIR}`: The directory for saving the annotation file, default to`'data/kitti2012/'`,
  and annotation files for train and test dataset will be save as `${SAVE_DIR}/KITTI2012_train.json`.

**Note**:

Annotation file is not required for local file storage, and it will be used in dataset config file when using cloud object storage like s3 storage. There is an example for using object storage:

```python
file_client_args= dict(
    backend='s3',
    path_mapping=dict(
        {'data/': 's3://dataset_path'}))
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', file_client_args=file_client_args, sparse=True)]
kitti_train = dict(
    type='KITTI2012',
    ann_file='KITTI2012_train.json',
    data_root='data/kitti2012',
    pipeline=kitti_train_pipeline,
    test_mode=False)
```
