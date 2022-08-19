# Prepare FlyingChairsOcc dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{Hur:2019:IRR,
  Author = {Junhwa Hur and Stefan Roth},
  Booktitle = {CVPR},
  Title = {Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation},
  Year = {2019}
}
```

## Download and Unpack dataset

Please download the datasets from the official websites.

```bash
wget https://download.visinf.tu-darmstadt.de/data/flyingchairs_occ/FlyingChairsOcc.tar.gz
tar -xvf FlyingChairsOcc.tar.gz
```

If your dataset folder structure is different from the following, you may need to change the corresponding paths.

```text

├── FlyingChairsOcc
|   ├── data
|   |    ├── xxxxx_flow.flo
|   |    ├── xxxxx_flow_b.flo
|   |    ├── xxxxx_img1.ppm
|   |    ├── xxxxx_img2.ppm
|   |    ├── xxxxx_occ1.png
|   |    ├── xxxxx_occ2.png
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.

```bash
python tools/dataset_converters/prepare_flyingchairsocc.py[optional arguments]
```

This scrip accepts these arguments:

- `--data-root ${DATASET_DIR}`: The dataset directory of FlyingChairsOcc, default to `'data/FlyingChairsOcc'`.

- `--save-dir ${SAVE_DIR}`: The directory for saving the annotation file, default to`'data/FlyingChairsOcc/'`,
  and annotation files for train and test dataset will be save as `${SAVE_DIR}/train.json` and `${SAVE_DIR}/test.json`

**Note**:

Annotation file is not required for local file storage, and it will be used in dataset config file when using cloud object storage like s3 storage. There is an example for using object storage:

```python
file_client_args= dict(
    backend='s3',
    path_mapping=dict(
        {'data/': 's3://dataset_path'}))
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_occ=True, file_client_args=file_client_args),
]
flyingchairsocc_train = dict(
    type='FlyingChairsOcc',
    ann_file='train.json',
    data_root='data/FlyingChairsOcc/',
    pipeline=train_pipeline,
    test_mode=False)
```
