# Tutorial 4: Train and test with existing models

Flow estimators pre-trained on the FlyingChairs and FlyingThings3d can serve as a good pre-trained model for other datasets.
This tutorial provides instruction for users to use the models provided in the [Model Zoo](../model_zoo.md) for other datasets to obtain better performance.
MMFlow also provides out-of-the-box tools for training models.
This section will show how to train and test _predefined_ models on standard datasets.

## Train models on standard datasets

### Modify training schedule

The fine-tuning hyper-parameters vary from the default schedule. It usually requires smaller learning rate and less training iterations.

```python
# training schedule for S_long schedule
train_cfg = dict(by_epoch=False, max_iters=1200000, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001, weight_decay=0.0004, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=False,
    gamma=0.5,
    milestones=[400000, 600000, 800000, 1000000])

# basic hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=100000, by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='FlowVisualizationHook'))
```

### Use pre-trained model

Users can load a pre-trained model by setting the `load_from` field of the config to the model's path or link.
The users might need to download the model weights before training to avoid the download time during training.

```python
# use the pre-trained model for the whole PWC-Net
load_from = 'https://download.openmmlab.com/mmflow/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.pth'  # model path can be found in model zoo
```

### Training on a single GPU

We provide `tools/train.py` to launch training jobs on a single GPU.
The basic usage is as follows.

```shell
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
```

During training, log files and checkpoints will be saved to the working directory, which is specified by `work_dir` in the config file or via CLI argument `--work-dir`.

This tool accepts several optional arguments, including:

- `--work-dir ${WORK_DIR}`: Override the working directory.
- `--amp`: Use auto mixed precision training.
- `--resume ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--cfg-options ${OVERRIDE_CONFIGS}`: Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.
  For example, '--cfg-option model.encoder.in_channels=6'. Please see this [guide](./1_config.md#Modify-config-through-script-arguments) for more details.

Below is the optional arguments for multi-gpu test:

- `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `--local_rank`: ID for local rank. If not specified, it will be set to 0.

**Note**:

Difference between `--resume` and `load-from`:

`--resume` loads both the model weights and optimizer status, and the iteration is also inherited from the specified checkpoint.
It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training iteration starts from 0. It is usually used for fine-tuning.

### Training on CPU

The process of training on the CPU is consistent with single GPU training. We just need to disable GPUs before the training process.

```shell
export CUDA_VISIBLE_DEVICES=-1
```

And then run the script [above](#training-on-a-single-GPU).

We do not recommend users to use CPU for training because it is too slow. We support this feature to allow users to debug on machines without GPU for convenience.

### Training on multiple GPUs

MMFlow implements **distributed** training with `MMDistributedDataParallel`.

We provide `tools/dist_train.sh` to launch training on multiple GPUs.
The basic usage is as follows.

```shell
sh tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

Optional arguments remain the same as stated [above](#training-on-a-single-gpu)
and has additional arguments to specify the number of GPUs.

#### Launch multiple jobs on a single machine

If you would like to launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/dist_train.sh ${CONFIG_FILE} 4
```

### Training on multiple nodes

MMFlow relies on `torch.distributed` package for distributed training.
Thus, as a basic usage, one can launch distributed training via PyTorch's [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

#### Train with multiple machines

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
```

Usually it is slow if you do not have high speed networking like InfiniBand.

#### Manage jobs with Slurm

[Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs. It supports both single-node and multi-node training.

The basic usage is as follows.

```shell
[GPUS=${GPUS}] sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

Below is an example of using 8 GPUs to train PWC-Net on a Slurm partition named _dev_, and set the work-dir to some shared file systems.

```shell
GPUS=8 sh tools/slurm_train.sh dev pwc_chairs configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py work_dir/pwc_chairs
```

You can check [the source code](../../../tools/dist_train.sh) to review full arguments and environment variables.

When using Slurm, the port option need to be set in one of the following ways:

1. Set the port through `--cfg-options`. This is more recommended since it does not change the original configs.

   ```shell
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --cfg-options env_cfg.dist_cfg.port=29500
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --cfg-options env_cfg.dist_cfg.port=29501
   ```

2. Modify the config files to set different communication ports.

   In `config1.py`, set

   ```python
   env_cfg = dict(dist_cfg=dict(backend='nccl', port=29500))
   ```

   In `config2.py`, set

   ```python
   env_cfg = dict(dist_cfg=dict(backend='nccl', port=29501))
   ```

   Then you can launch two jobs with `config1.py` and `config2.py`.

   ```shell
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```

## Test models on standard datasets

We provide testing scripts for evaluating an existing model on the whole dataset.
The following testing environments are supported:

- single GPU
- CPU
- single node multiple GPUs
- multiple nodes

Choose the proper script to perform testing depending on the testing environment.
It should be pointed that only FlownetS, GMA and RAFT support testing on CPU.

```shell
# single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--work-dir ${WORK_DIR}] \
    [--show ${SHOW_FLOW}] \
    [--show-dir ${VISUALIZATION_DIRECTORY}] \
    [--wait-time ${SHOW_INTERVAL}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]

# CPU testing
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--work-dir ${WORK_DIR}] \
    [--show ${SHOW_FLOW}] \
    [--show-dir ${VISUALIZATION_DIRECTORY}] \
    [--wait-time ${SHOW_INTERVAL}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]

# multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--work-dir ${WORK_DIR}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]
```

`tools/dist_test.sh` also supports multi-node testing, but relies on PyTorch's [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

[Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_test.sh` to spawn testing jobs. It supports both single-node and multi-node testing.

```shell
[GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} \
    ${CONFIG_FILE} ${CHECKPOINT_FILE} \
    [--work-dir ${OUTPUT_DIRECTORY}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]
```

Optional arguments:

- `--work-dir`: If specified, results will be saved in this directory. If not specified, the results will be automatically saved to `work_dirs/{CONFIG_NAME}`.
- `--show`: Show prediction results at runtime, available when `--show-dir` is not specified.
- `--show-dir`: If specified, the visualized optical flow map will be saved in the specified directory.
- `--wait-time`: The interval of show (s), which takes effect when `--show` is activated. Default to 2.
- `--cfg-options`:  If specified, the key-value pair in xxx=yyy format will be merged into config file.
  For example, '--cfg-option model.encoder.in_channels=6'. Please see this [guide](./1_config.md#Modify-config-through-script-arguments) for more details.

Below is the optional arguments for multi-gpu test:

- `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `--local_rank`: ID for local rank. If not specified, it will be set to 0.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`,
and test PWC-Net on FlyingChairs without saving predicted flow files. The basic usage is as follows.

```shell
python tools/test.py configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py \
    checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth
```

Since `--work-dir` is not specified, the folder `work_dirs/pwcnet_8x1_slong_flyingchairs_384x448` will be created automatically to save the evaluation results.

If you want to show the predicted optical flow at runtime, just run

```shell
python tools/test.py configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py \
    checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth --show
```

Every image shown consists of two images, the ground truth on the left and the prediction result on the right.
The image will be shown for 2 seconds, you can adjust `--wait-time` to change the display time.
According to the default setting, the results are show every 50 results.
If you want to change the frequency, for example, you want every result to be shown,
then add `--cfg-options default_hooks.visualization.interval=1` to the above command.
Of course, you can also modify the relevant parameters in config files.
For more details of visualization, please see this [guide](./visualization.md).

If you want to save the predicted optical flow, just specify the `--show-dir`.
For example, if we want to save the predicted results in `show_dirs`, then run

```shell
python tools/test.py configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py \
    checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth --show-dir show_dirs
```

Similarly, you can change the frequency of saving results by the above method.

We recommend using single gpu and setting batch_size=1 to evaluate models, as it must ensure that the number of dataset samples
can be divisible by batch size, so even if working on slurm, we will use one gpu to test.
Assume our partition is Test and job name is test_pwc, so here is the example:

```shell
GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=2 ./tools/slurm_test.sh Test test_pwc \
    configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py \
    checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth
```
