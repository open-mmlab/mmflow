# Tutorial 3: Inference with existing models

MMFlow provides pre-trained models for flow estimation in [Model Zoo](../model_zoo.md), and supports multiple standard datasets, including FlyingChairs, Sintel, etc. This note will show how to perform common tasks on these existing models and standard datasets, including:

- Use existing models to inference on given images.
- Test existing models on standard datasets.

## Inference on given images

MMFlow provides high-level Python APIs for inference on images. Here is an example of building the model and inference on given images.
Please download the [pre-trained model](https://download.openmmlab.com/mmflow/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.pth) to the path specified by `checkpoint_file` first.

```python
from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow, write_flow
from mmflow.utils import register_all_modules

# Specify the path to model config and checkpoint file
config_file = 'configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py'
checkpoint_file = 'checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth'

# register all modules in mmflow into the registries
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test image pair, and save the results
img1 = 'demo/frame_0001.png'
img2 = 'demo/frame_0002.png'
result = inference_model(model, img1, img2)

# The original `result` is a list, and the elements inside are of the `FlowDataSample` data type
# get prediction from result and convert to np
result = result[0].pred_flow_fw.data.permute(1, 2, 0).cpu().numpy()

# save the optical flow file
write_flow(result, flow_file='flow.flo')

# save the visualized flow map
visualize_flow(result, save_file='flow_map.png')
```

An image demo can be found in [demo/image_demo.py](../../demo/image_demo.py).

## Evaluate existing models on standard datasets

### Test existing models

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
    [--work-dir ${OUTPUT_DIRECTORY}] \
    [--show ${SHOW_FLOW}] \
    [--show-dir ${VISUALIZATION_DIRECTORY}] \
    [--wait-time ${SHOW_INTERVAL}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]

# CPU
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--work-dir ${OUTPUT_DIRECTORY}] \
    [--show ${SHOW_FLOW}] \
    [--show-dir ${VISUALIZATION_DIRECTORY}] \
    [--wait-time ${SHOW_INTERVAL}] \
    [--cfg-options ${OVERRIDE_CONFIGS}]

# multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--work-dir ${OUTPUT_DIRECTORY}] \
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
- `--cfg-options`:  If specified, the key-value pair optional cfg will be merged into config file.
  For example, '--cfg-option model.encoder.in_channels=6'. Please see this [guide](./1_config.md) for more details.

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

Similarly, you can also change the frequency of saving results by the above method.

We recommend using single gpu and setting batch_size=1 to evaluate models, as it must ensure that the number of dataset samples
can be divisible by batch size, so even if working on slurm, we will use one gpu to test.
Assume our partition is Test and job name is test_pwc, so here is the example:

```shell
GPUS=1 GPUS_PER_NODE=1 CPUS_PER_TASK=2 ./tools/slurm_test.sh Test test_pwc \
    configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py \
    checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth
```
