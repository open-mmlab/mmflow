# Prepare hd1k dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{kondermann2016hci,
  title={The HCI Benchmark Suite: Stereo and Flow Ground Truth With Uncertainties for Urban Autonomous Driving},
  author={Kondermann, Daniel and Nair, Rahul and Honauer, Katrin and Krispin, Karsten and Andrulis, Jonas and Brock, Alexander and Gussefeld, Burkhard and Rahimimoghaddam, Mohsen and Hofmann, Sabine and Brenner, Claus and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={19--28},
  year={2016}
}
```

## Download and Unpack dataset

You can download datasets on this [webpage](http://hci-benchmark.iwr.uni-heidelberg.de/). Then, you need to unzip and move corresponding datasets to follow the folder structure shown below. The datasets have been well-prepared by the original authors.

```text
├── hd1k
|    ├── hd1k_flow_gt
|    |    ├── flow_occ
|    |    |     ├── xxxxxx_xxxx.png
|    ├── hd1k_input
|    |    ├── image_2
|    |    |     ├── xxxxxx_xxxx.png
```

## Generate annotation file

We provide a convenient script to generate annotation file, which list all of data samples in the dataset.
You can use the following command to generate annotation file.

```bash
python tools/dataset_converters/prepare_hd1k.py [optional arguments]
```

This scrip accepts these arguments:

- `--data-root ${DATASET_DIR}`: The dataset directory of FlyingChairs, default to `'data/hd1k'`.

- `--save-dir ${SAVE_DIR}`: The directory for saving the annotation file, default to`'data/hd1k/'`,
  and annotation files for train and test dataset will be save as `${SAVE_DIR}/train.json`.

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
hd1k_train = dict(
    type='HD1K',
    ann_file='train.json', # train.json is in data_root i.e. data/hd1k/
    pipeline=train_pipeline,
    data_root='data/hd1k',
    test_mode=False)
```
