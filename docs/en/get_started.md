# Get Started

This page provides basic tutorials about the usage of MMFlow.
For installation instructions, please see [install.md](install.md).

<!-- TOC -->

- [Get Started](#get-started)
  - [Prepare datasets](#prepare-datasets)
  - [Inference with Pre-trained Models](#inference-with-pre-trained-models)
    - [Run a demo](#run-a-demo)
    - [Test a dataset](#test-a-dataset)
  - [Train a model](#train-a-model)
  - [Tutorials](#tutorials)

<!-- TOC -->

## Prepare datasets

It is recommended to symlink the dataset root to `$MMFlow/data`.
Please follow the corresponding guidelines for data preparation.

- [FlyingChairs](data_prepare/FlyingChairs/README.md)
- [FlyingThings3d_subset](data_prepare/FlyingThings3d_subset/README.md)
- [FlyingThings3d](data_prepare/FlyingThings3d/README.md)
- [Sintel](data_prepare/Sintel/README.md)
- [KITTI2015](data_prepare/KITTI2015/README.md)
- [KITTI2012](data_prepare/KITTI2012/README.md)
- [FlyingChairsOcc](data_prepare/FlyingChairsOcc/README.md)
- [ChairsSDHom](data_prepare/ChairsSDHom/README.md)
- [HD1K](data_prepare/hd1k/README.md)

## Inference with Pre-trained Models

We provide testing scripts to evaluate a whole dataset (Sintel, KITTI2015, etc.),
and provide some high-level APIs and scripts to estimate flow for images or a video easily.

### Run a demo

We provide scripts to run demos. Here is an example to predict the optical flow between two adjacent frames.

1. [image demo](../demo/image_demo.py)

   ```shell
   python demo/image_demo.py ${IMAGE1} ${IMAGE2} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUTPUT_DIR} \
       [--out_prefix] ${OUTPUT_PREFIX} [--device] ${DEVICE}
   ```

   Optional arguments:

   - `--out_prefix`: The prefix for the output results including flow file and visualized flow map.
   - `--device`: Device used for inference.

   Example:

   Assume that you have already downloaded the checkpoints to the directory `checkpoints/`,
   and output will be saved in the directory `raft_demo`.

   ```shell
   python demo/image_demo.py demo/frame_0001.png demo/frame_0002.png \
       configs/raft/raft_8x2_100k_mixed_368x768.py \
       checkpoints/raft_8x2_100k_mixed_368x768.pth raft_demo
   ```

2. [video demo](../demo/video_demo.py)

   ```shell
   python demo/video_demo.py ${VIDEO} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${OUTPUT_FILE} \
       [--gt] ${GROUND_TRUTH} [--device] ${DEVICE}
   ```

   Optional arguments:

   - `--gt`: The video file of ground truth for input video.
     If specified, the ground truth will be concatenated predicted result as a comparison.
   - `--device`: Device used for inference.

   Example:

   Assume that you have already downloaded the checkpoints to the directory `checkpoints/`,
   and output will be save as `raft_demo.mp4`.

   ```shell
   python demo/video_demo.py demo/demo.mp4 \
       configs/raft/raft_8x2_100k_mixed_368x768.py \
       checkpoints/raft_8x2_100k_mixed_368x768.pth \
       raft_demo.mp4 --gt demo/demo_gt.mp4
   ```

### Test a dataset

You can use the following commands to test a dataset, and more information is in [tutorials/1_inference](tutorials/1_inference.md).

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Optional arguments:

- `--out_dir`: Directory to save the output results. If not specified, the flow files will not be saved.
- `--fuse-conv-bn`: Whether to fuse conv and bn, this will slightly increase the inference speed.
- `--show_dir`: Directory to save the visualized flow maps. If not specified, the flow maps will not be saved.
- `--eval`: Evaluation metrics, e.g., "EPE".
- `--cfg-option`: Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.
  For example, '--cfg-option model.encoder.in_channels=6'.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

Test PWC-Net on Sintel clean and final sub-datasets without saving predicted flow files and evaluating the EPE.

```shell
python tools/test.py configs/pwcnet/pwcnet_ft_4x1_300k_sintel_384x768.py \
    checkpoints/pwcnet_8x1_sfine_sintel_384x768.pth --eval EPE
```

## Train a model

You can use the [train script](../tools/train.py) to launch training task with a single GPU,
and more information in [tutorials/2_finetune](tutorials/2_finetune.md)

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Optional arguments:

- `--work-dir`: Override the working directory specified in the config file.
- `--load-from`: The checkpoint file to load weights from.
- `--resume-from`: Resume from a previous checkpoint file.
- `--no-validate`: Whether not to evaluate the checkpoint during training.
- `--seed`: Seed id for random state in python, numpy and pytorch to generate random numbers.
- `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
- `--cfg-options`: Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.
  For example, '--cfg-option model.encoder.in_channels=6'.

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch/iter is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch/iter starts from 0. It is usually used for finetuning.

Here is an example to train PWC-Net.

```shell
python tools/train.py configs/pwcnet/pwcnet_ft_4x1_300k_sintel_384x768.py --work-dir work_dir/pwcnet
```

## Tutorials

We provide some tutorials for users:

- [learn about configs](tutorials/0_config.md)
- [inference model](tutorials/1_inference.md)
- [finetune model](tutorials/2_finetune.md)
- [customize data pipelines](tutorials/3_data_pipeline.md)
- [add new modules](tutorials/4_new_modules.md)
- [customize runtime settings](tutorials/5_customize_runtime.md).
