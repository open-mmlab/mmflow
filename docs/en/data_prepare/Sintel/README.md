# Prepare Sintel dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{Butler:ECCV:2012,
title = {A naturalistic open source movie for optical flow evaluation},
author = {Butler, D. J. and Wulff, J. and Stanley, G. B. and Black, M. J.},
booktitle = {European Conf. on Computer Vision (ECCV)},
editor = {{A. Fitzgibbon et al. (Eds.)}},
publisher = {Springer-Verlag},
series = {Part IV, LNCS 7577},
month = oct,
pages = {611--625},
year = {2012}
}

@inproceedings{Wulff:ECCVws:2012,
 title = {Lessons and insights from creating a synthetic optical flow benchmark},
 author = {Wulff, J. and Butler, D. J. and Stanley, G. B. and Black, M. J.},
 booktitle = {ECCV Workshop on Unsolved Problems in Optical Flow and Stereo Estimation},
 editor = {{A. Fusiello et al. (Eds.)}},
 publisher = {Springer-Verlag},
 series = {Part II, LNCS 7584},
 month = oct,
 pages = {168--177},
 year = {2012}
}
```

## Download and Unpack dataset

Please download the datasets from the official websites.

```bash
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
# or use US mirror wget http://sintel.cs.washington.edu/MPI-Sintel-complete.zip
unzip MPI-Sintel-complete.zip
```

If your dataset folder structure is different from the following, you may need to change the corresponding paths.

```text
├── Sintel
|   ├── training
|   |   ├── clean
|   |   |   ├── xxxx_x
|   |   |   |    ├── frame_xxxx.png
|   |   ├── final
|   |   |   ├── xxxx_x
|   |   |   |    ├── frame_xxxx.png
|   |   ├── flow
|   |   |   |    ├── frame_xxxx.flo
|   |   ├── invalid
|   |   |   ├── xxxx_x
|   |   |   |    ├── frame_xxxx.png
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.

```bash
python tools/dataset_converters/prepare_sintel.py [optional arguments]
```

This script accepts these arguments:

- `--data-root ${DATASET_DIR}`: The dataset directory of Sintel, default to `'data/Sintel'`.

- `--save-dir ${SAVE_DIR}`: The directory for saving the annotation file, default to`'data/Sintel/'`,
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
    dict(type='LoadAnnotations', backend_args=backend_args)]
sintel_clean_train = dict(
    type='Sintel',
    ann_file='train.json', # train.json is in data_root i.e. data/Sintel/
    pipeline=train_pipeline,
    data_root='data/Sintel',
    test_mode=False,
    pass_style='clean')
```
