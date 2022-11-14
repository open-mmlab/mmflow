# Prepare ChairsSDHom dataset

<!-- [DATASET] -->

```bibtex
@InProceedings{IMKDB17,
  author    = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
  title     = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
  booktitle = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
  month     = "Jul",
  year      = "2017",
  url       = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}
```

## DownLoad and unpack dataset

Please download the datasets from the official websites.

```bash
wget https://lmb.informatik.uni-freiburg.de/data/FlowNet2/ChairsSDHom/ChairsSDHom.tar.gz
tar -xvf ChairsSDHom.tar.gz
```

If your dataset folder structure is different from the following, you may need to change the corresponding paths.

```text
ChairsSDHom
|   ├── data
|   |    ├── train
|   |    |    |── flow
|   |    |    |      |── xxxxx.pfm
|   |    |    |── t0
|   |    |    |      |── xxxxx.png
|   |    |    |── t1
|   |    |    |      |── xxxxx.png
|   |    ├── test
|   |    |    |── flow
|   |    |    |      |── xxxxx.pfm
|   |    |    |── t0
|   |    |    |      |── xxxxx.png
|   |    |    |── t1
|   |    |    |      |── xxxxx.png
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.

```bash
python tools/dataset_converters/prepare_chairssdhom.py [optional arguments]
```

This script accepts these arguments:

- `--data-root ${DATASET_DIR}`: The dataset directory of ChairsSDHom, default to `'data/ChairsSDHom'`.

- `--save-dir ${SAVE_DIR}`: The directory for saving the annotation file, default to`'data/ChairsSDHom/'`,
  and annotation files for train and test dataset will be save as `${SAVE_DIR}/train.json` and `${SAVE_DIR}/test.json`

**Note**:

Annotation file is not required for local file storage, and it will be used in dataset config file when using cloud object storage like s3 storage. There is an example for using object storage:

```python
file_client_args = dict(
    backend='s3',
    path_mapping=dict(
        {'data/': 's3://dataset_path'}))
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', file_client_args=file_client_args)]
chairssdhom_train = dict(
    type='ChairsSDHom',
    ann_file='train.json', # train.json is in data_root i.e. data/ChairsSDHom/
    pipeline=train_pipeline,
    data_root='data/ChairsSDHom')
```
