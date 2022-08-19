# Prepare FlyingThing3d dataset

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

You can download datasets via \[BitTorrent\] (https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar.torrent). Then, you need to unzip and move corresponding datasets to follow the folder structure shown below. The datasets have been well-prepared by the original authors.

```text
├── flyingthings3d
|   ├── frames_cleanpass
|   |   ├── TEST
|   |   |   ├── x
|   |   |   |   ├── xxxx
|   |   |   |   |    ├── left
|   |   |   |   |    |   ├── xxxx.png
|   |   |   |   |    ├── right
|   |   |   |   |    |   ├── xxxx.png
|   |   ├── TRAIN
|   |   |   ├── x
|   |   |   |   ├── xxxx
|   |   |   |   |    ├── left
|   |   |   |   |    |   ├── xxxx.png
|   |   |   |   |    ├── right
|   |   |   |   |    |   ├── xxxx.png
|   ├── frames_finalpass
|   |   ├── TEST
|   |   |   ├── x
|   |   |   |   ├── xxxx
|   |   |   |   |    ├── left
|   |   |   |   |    |   ├── xxxx.png
|   |   |   |   |    ├── right
|   |   |   |   |    |   ├── xxxx.png
|   |   ├── TRAIN
|   |   |   ├── x
|   |   |   |   ├── xxxx
|   |   |   |   |    ├── left
|   |   |   |   |    |   ├── xxxx.png
|   |   |   |   |    ├── right
|   |   |   |   |    |   ├── xxxx.png
|   ├── optical_flow
|   |   ├── TEST
|   |   |   ├── x
|   |   |   |   |   ├── xxxx
|   |   |   |   |    ├── into_future
|   |   |   |   |    |       ├── left
|   |   |   |   |    |       |     ├── OpticalFlowIntoFuture_xxxx_L.pfm
|   |   |   |   |    |       ├── right
|   |   |   |   |    |       |     ├── OpticalFlowIntoFuture_xxxx_R.pfm
|   |   |   |   |    ├── into_past
|   |   |   |   |    |       ├── left
|   |   |   |   |    |       |     ├── OpticalFlowIntoPast_xxxx_L.pfm
|   |   |   |   |    |       ├── right
|   |   |   |   |    |       |     ├── OpticalFlowIntoPast_xxxx_R.pfm
|   |   ├── TRAIN
|   |   |   ├── x
|   |   |   |   ├── xxxx
|   |   |   |   |    ├── into_future
|   |   |   |   |    |       ├── left
|   |   |   |   |    |       |     ├── OpticalFlowIntoFuture_xxxx_L.pfm
|   |   |   |   |    |       ├── right
|   |   |   |   |    |       |     ├── OpticalFlowIntoFuture_xxxx_R.pfm
|   |   |   |   |    ├── into_past
|   |   |   |   |    |       ├── left
|   |   |   |   |    |       |     ├── OpticalFlowIntoPast_xxxx_L.pfm
|   |   |   |   |    |       ├── right
|   |   |   |   |    |       |     ├── OpticalFlowIntoPast_xxxx_R.pfm
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.

```bash
python tools/dataset_converters/prepare_flyingthings3d.py [optional arguments]
```

This scrip accepts these arguments:

- `--data-root ${DATASET_DIR}`: The dataset directory of FlyingThings3D, default to `'data/flyingthings3d'`.

- `--save-dir ${SAVE_DIR}`: The directory for saving the annotation file, default to`'data/flyingthings3d/'`,
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
    dict(type='LoadAnnotations', file_client_args=file_client_args),
]
flyingthings3d_train_cleanpass = dict(
    type='FlyingThings3D',
    ann_file='train.json',
    pipeline=train_pipeline,
    data_root='data/flyingthings3d',
    test_mode=False,
    pass_style='clean',
    scene='left',
    double=True)
```
