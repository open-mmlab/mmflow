# Tutorial 1: Inference with existing models

MMFlow provides pre-trained models for flow estimation in [Model Zoo](../model_zoo.md), and supports multiple standard datasets, including FlyingChairs, Sintel, etc. This note will show how to perform common tasks on these existing models and standard datasets, including:

- Use existing models to inference on given images.
- Test existing models on standard datasets.

## Inference on given images

MMFlow provides high-level Python APIs for inference on images. Here is an example of building the model and inference on given images.

```python
from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow, write_flow
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py'
checkpoint_file = 'checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test image pair, and save the results
img1='demo/frame_0001.png'
img2='demo/frame_0002.png'
result = inference_model(model, img1, img2)
# save the optical flow file
write_flow(result, flow_file='flow.flo')
# save the visualized flow map
flow_map = visualize_flow(result, save_file='flow_map.png')
```

An image demo can be found in [demo/image_demo.py](../../../demo/image_demo.py).

## Evaluate existing models on standard datasets

### Test existing models

We provide testing scripts for evaluating an existing model on the whole dataset.
The following testing environments are supported:

- single GPU
- CPU
- single node multiple GPUs
- multiple nodes

Choose the proper script to perform testing depending on the testing environment.

```shell
# single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--eval ${EVAL_METRICS}] \
    [--out-dir ${OUTPUT_DIRECTORY}] \
    [--show-dir ${VISUALIZATION_DIRECTORY}]

# CPU: disable GPUs and run single-gpu testing script
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \
    [--show]

# multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--eval ${EVAL_METRICS}] \
    [--out-dir ${OUTPUT_DIRECTORY}]
```

`tools/dist_test.sh` also supports multi-node testing, but relies on PyTorch's [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility).

[Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.
On a cluster managed by Slurm, you can use `slurm_test.sh` to spawn testing jobs. It supports both single-node and multi-node testing.

```shell
[GPUS=${GPUS}] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} \
    ${CONFIG_FILE} ${CHECKPOINT_FILE} \
    [--eval ${EVAL_METRICS}] \
    [--out-dir ${OUTPUT_DIRECTORY}]
```

Optional arguments:

- `--eval`: Evaluation metrics, e.g., "EPE".
- `--fuse-conv-bn`: Whether to fuse conv and bn, this will slightly increase the inference speed.
- `--out-dir`: If specified, predicted optical flow will be saved in this directory.
- `--show-dir`: if specified, the visualized optical flow map will be saved in this directory.
- `--cfg-options`:  If specified, the key-value pair optional cfg will be merged into config file.
  For example, '--cfg-option model.encoder.in_channels=6'.

Below is the optional arguments for multi-gpu test:

- `--gpu_collect`: If specified, recognition results will be collected using gpu communication. Otherwise, it will save the results on different gpus to `TMPDIR` and collect them by the rank 0 worker.
- `--tmpdir`: Temporary directory used for collecting results from multiple workers, available when `--gpu_collect` is not specified.
- `--launcher`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `--local_rank`: ID for local rank. If not specified, it will be set to 0.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`,
and test PWC-Net on Sintel clean and final sub-datasets without save predicted flow files and evaluate the EPE.

```shell
python tools/test.py configs/pwc_net_8x1_sfine_sintel_384x768.py \
    checkpoints/pwcnet_8x1_sfine_sintel_384x768.pth --eval EPE
```

We recommend using single gpu and setting batch_size=1 to evaluate models, as it must ensure that the number of dataset samples
can be divisible by batch size, so even if working on slurm, we will use one gpu to test.
Assume our partition is Test and job name is test_pwc, so here is the example:

```shell
GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=2 ./tools/slurm_test.sh Test test_pwc \
    configs/pwc_net_8x1_sfine_sintel_384x768.py \
    checkpoints/pwcnet_8x1_sfine_sintel_384x768.pth --eval EPE
```
