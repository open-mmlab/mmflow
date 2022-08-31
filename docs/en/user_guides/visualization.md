# Visualization

MMFlow 1.x provides convenient ways for monitoring training status or visualizing data and model predictions.

## Training status Monitor

MMFlow 1.x uses TensorBoard to monitor training status.

### TensorBoard Configuration

Install TensorBoard following [official instructions](https://www.tensorflow.org/install)

```shell
pip install future tensorboard
```

Add `TensorboardVisBackend` in `vis_backend` of `visualizer` in `configs/_base_/default_runtime.py`:

```python
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='FlowLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

The configuration contains `LocalVisBackend`, which means the scalars during training will be stored locally as well.

### Examining scalars in TensorBoard

Launch training experiment e.g.

```shell
python tools/train.py configs/pwcnet/pwcnet_8xb1_slong_flyingchairs-384x448.py --work-dir work_dirs/test_visual
```

You can specify the `save_dir` in `visualizer` to modify the storage path.
The default storage path is `vis_data` under your `work_dir`.
For example, the `vis_data` path of a particular experiment is

```shell
work_dirs/test_visual/20220831_165919/vis_data
```

The scalar file in `vis_data` includes learning rate, losses and data_time etc, and also record metrics results during evaluation.
You can refer to [logging tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/logging.html) in mmengine to log custom data.
The TensorBoard visualization results are executed with the following command:

```shell
tensorboard --logdir work_dirs/test_visual/20220831_165919/vis_data
```

## Prediction Visualization

MMFlow provides `FlowVisualizationHook` that can render optical flow of ground truth and prediction.
Users can modify `visualization` in `default_hooks` to invoke the hook.
MMFlow configures `default_hooks` in each file under `configs/_base_/schedules`.
For example, in `configs/_base_/schedules/schedules_s_long.py`, let's modify the `FlowVisualizationHook` related parameters.
Set `draw` to `True` to enable the storage of network inference results.
`interval` indicates the sampling interval of the predicted results, defaults to 50, and when set to 1, each inference result of the network will be saved.

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

Additionally, if you want to keep the original file under `configs` unchanged, you can specify `--cfg-options` in commands by referring to this [guide](https://github.com/open-mmlab/mmflow/blob/dev-1.x/docs/en/user_guides/1_config.md#modify-config-through-script-arguments).

```shell
python tools/test.py \
    configs/pwcnet/pwcnet_8xb1_slong_flyingchairs-384x448.py \
    work_dirs/download/hub/checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth \
    --work-dir work_dirs/test_visual \
    --cfg-options default_hooks.visualization.draw=True default_hooks.visualization.interval=1
```

The default backend of visualization is `LocalVisBackend`, which means storing the visualization results locally.
Backend related configuration is in `configs/_base_/default_runtime.py`.
In order to enable TensorBoard visualization as well, modify the `visulizer` just as this [configuration](https://github.com/open-mmlab/mmflow/blob/dev-1.x/docs/en/user_guides/visualization.md#tensorboard-configuration).
Assume the `vis_data` path of a particular test is

```shell
work_dirs/test_visual/20220831_114424/vis_data
```

The stored results of the local visualization are kept in `vis_image` under `vis_data`, while the TensorBoard visualization results can be executed with the following command:

```shell
tensorboard --logdir work_dirs/test_visual/20220831_114424/vis_data
```

The visualization image consists of two parts, the ground truth on the left and the network prediction result on the right.
