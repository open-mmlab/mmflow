# Prepare FlyingChairs dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{DFIB15,
  author    = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz{\i}rba{\c{s}} and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
  title     = "FlowNet: Learning Optical Flow with Convolutional Networks",
  booktitle = "IEEE International Conference on Computer Vision (ICCV)",
  month     = " ",
  year      = "2015",
  url       = "http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15"
}
```

## Download and Unpack dataset

Please download the datasets from the official websites.

```bash
wget https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip
unzip FlyingChairs.zip
cd FlyingChairs_release
wget https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt
```

If your dataset folder structure is different from the following, you may need to change the corresponding paths.

```text

├── FlyingChairs_release
│   ├── FlyingChairs_train_val.txt
|   ├── data
|   |    ├── xxxxx_flow.flo
|   |    ├── xxxxx_img1.ppm
|   |    ├── xxxxx_img2.ppm
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.

```bash
python tools/dataset_converters/prepare_flyingchairs.py  [optional arguments]
```

This scrip accepts these arguments:

- `--data-root ${DATASET_DIR}`: The dataset directory of FlyingChairs, default to `'data/FlyingChair_release'`.

- `--split-file ${SPLIT_FILE}`: The file for splitting train and test dataset, default to `'data/FlyingChairs_release/FlyingChairs_train_val.txt'`.

- `--save-dir ${SAVE_DIR}`: The directory for saving the annotation file, default to`'data/FlyingChairs_release/'`,
  and annotation files for train and test dataset will be save as `${SAVE_DIR}/train.json` and `${SAVE_DIR}/test.json`

**Note**:

Annotation file is not required for local file storage, and it will be used in dataset config file when using cloud object storage like s3 storage. There is an example for using object storage:

```python
dataset_type = 'FlyingChairs'
data_root = 'data/FlyingChairs_release'
file_client_args= dict(
    backend='s3',
    path_mapping=dict(
        {'data/': 's3://dataset_path'}))
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', file_client_args=file_client_args),
]
flyingchairs_train = dict(
    type=dataset_type,
    ann_file='train.json',
    pipeline=train_pipeline,
    data_root=data_root)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', file_client_args=file_client_args),
]
flyingchairs_test = dict(
    type=dataset_type,
    ann_file='test.json',
    pipeline=test_pipeline,
    data_root=data_root,
    test_mode=True)
```
