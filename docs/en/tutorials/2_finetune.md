# Tutorial 2: Finetuning Models

Flow estimators pre-trained on the FlyingChairs and FlyingThings3d can serve as a good pre-trained model for other datasets.
This tutorial provides instruction for users to use the models provided in the [Model Zoo](../model_zoo.md) for other datasets to obtain better performance.
MMFlow also provides out-of-the-box tools for training models.
This section will show how to train _predefined_ models on standard datasets.

## Modify training schedule

The fine-tuning hyper-parameters vary from the default schedule. It usually requires smaller learning rate and less training iterations.

```python
# optimizer
optimizer = dict(type='Adam', lr=1e-5, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    gamma=0.5,
    step=[
        45000, 65000, 85000, 95000, 97500, 100000, 110000, 120000, 130000,
        140000
    ])
runner = dict(type='IterBasedRunner', max_iters=150000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='EPE')
```

## Use pre-trained model

Users can load a pre-trained model by setting the `load_from` field of the config to the model's path or link.
The users might need to download the model weights before training to avoid the download time during training.

```python
# use the pre-trained model for the whole PWC-Net
load_from = 'https://download.openmmlab.com/mmflow/pwcnet/pwcnet_8x1_sfine_flyingthings3d_subset_384x768.pth'  # model path can be found in model zoo
```

## Training on a single GPU

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
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--cfg-option`: Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.
  For example, '--cfg-option model.encoder.in_channels=6'.

**Note**:

Difference between `resume-from` and `load-from`:

`resume-from` loads both the model weights and optimizer status, and the iteration is also inherited from the specified checkpoint.
It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training iteration starts from 0. It is usually used for finetuning.

### Training on CPU

The process of training on the CPU is consistent with single GPU training. We just need to disable GPUs before the training process.

```shell
export CUDA_VISIBLE_DEVICES=-1
```

And then run the script [above](#training-on-a-single-GPU).

We do not recommend users to use CPU for training because it is too slow. We support this feature to allow users to debug on machines without GPU for convenience.

## Training on multiple GPUs

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

### Launch multiple jobs on a single machine

If you would like to launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 sh tools/dist_train.sh ${CONFIG_FILE} 4
```

## Training on multiple nodes

MMFlow relies on `torch.distributed` package for distributed training.
Thus, as a basic usage, one can launch distributed training via PyTorch's [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

### Train with multiple machines

If you launch with multiple machines simply connected with ethernet, you can simply run following commands:

On the first machine:

On the first machine:

```shell
NNODES=2 NODE_RANK=0 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
```

On the second machine:

```shell
NNODES=2 NODE_RANK=1 PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} sh tools/dist_train.sh ${CONFIG_FILE} ${GPUS}
```

Usually it is slow if you do not have high speed networking like InfiniBand.

### Manage jobs with Slurm

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

You can check [the source code](../../tools/dist_train.sh) to review full arguments and environment variables.

When using Slurm, the port option need to be set in one of the following ways:

1. Set the port through `--cfg-options`. This is more recommended since it does not change the original configs.

   ```shell
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --cfg-options 'dist_params.port=29500'
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --cfg-options 'dist_params.port=29501'
   ```

2. Modify the config files to set different communication ports.

   In `config1.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29500)
   ```

   In `config2.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29501)
   ```

   Then you can launch two jobs with `config1.py` and `config2.py`.

   ```shell
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   GPUS=4 GPUS_PER_NODE=4 sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```
