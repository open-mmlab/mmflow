# Visualization

MMFlow provides visualization hook, used to visualize validation and testing process prediction results.

## Usage

Users can modify `visualization` field in `default_hooks`. MMFlow configures `default_hooks` in each file under `configs/_base_/schedules`. For example, in `configs/_base_/schedules/schedules_s_long.py`, let's modify the `FlowVisualizationHook` related parameters. Set `draw` to `True` to enable the storage of network inference results. `interval` indicates the sampling interval of the predicted results, defaults to 50, and when set to 1, each inference result of the network will be saved. Now, the configuration of `default_hooks` is as follows:

```python
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=100000, by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='FlowVisualizationHook', draw=True, interval=1))
```

There is a way not to change the files under `configs/_base_`. For example, in `configs/pwcnet/pwcnet_8xb1_slong_flyingchairs-384x448.py` inherited from `configs/_base_/schedules/schedules_s_long.py`, just add `visualization` field in this way:

```python
default_hooks = dict(
    visualization=dict(type='FlowVisualizationHook', draw=True, interval=1))
```

Additionally, if you don't want to change the original files under `configs`, you can specify `--cfg-options` in commands by referring to this [guide](./1_config.md#modify-config-through-script-arguments).

```shell
python tools/test.py \
    configs/pwcnet/pwcnet_8xb1_slong_flyingchairs-384x448.py \
    work_dirs/download/hub/checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth \
    --work-dir work_dirs/test_visual \
    --cfg-options default_hooks.visualization.draw=True default_hooks.visualization.interval=1
```

The default backend of visualizatin is `LocalVisBackend`, which means storing the visualization results locally.
Backend related configuration is in `configs/_base_/default_runtime.py`.
In order to enable tensorboard visualization as well, modify the `visulizer` field in this way:

```python
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='FlowLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

You can change the storage path by specifying `save_dir` field in `visualizer`.
The default storage path for visualization result is the `vis_data` path under your `work_dir`.
For example, the `vis_data` path of a particular test is `work_dirs/test_visual/20220831_114424/vis_data`.
The stored results of the local visualization are kept in `vis_image` under `vis_data`, while the tensorboard visualization results can be executed with the following command:

```shell
tensorboard --logdir work_dirs/test_visual/20220831_114424/vis_data
```

The visualization image consists of two parts, the ground truth on the left and the network prediction result on the right.
