# Runtime Settings Customization

In this tutorial, we will introduce some methods about how to customize optimization methods, training schedules and hooks when running your own settings for the project.

## Customize optimization settings

Optimization related configuration is now all managed by OptimWrapper, which is a high-level API of optimizer. The OptimWrapper supports different training strategies, including auto mixed precision training, gradient accumulation and gradient clipping. `optim_wrapper` usually has three fields: `optimizer`, `paramwise_cfg`, `clip_grad`, refer to [OptimWrapper](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.md) for more detail. See the example below, where `Adam` is used as an optimizer and gradient clipping is added.

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001, weight_decay=0.0004, betas=(0.9, 0.999)))
    clip_grad=dict(max_norm=0.01, norm_type=2)
```

### Customize optimizer supported by PyTorch

We already support to use all the optimizers implemented by PyTorch, and the only modification is to change the `optimizer` field of config files.
For example, if you want to use `Adam`, the modification could be as the following.

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

To modify the learning rate of the model, the users only need to modify the `lr` in the config of optimizer.
The users can directly set arguments following the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

For example, if you want to use `Adam` with the setting like `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)` in PyTorch,
the modification could be set as the following.

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

### Customize self-implemented optimizer

#### 1. Define a new optimizer

A customized optimizer could be defined as following.

Assume you want to add an optimizer named `MyOptimizer`, which has arguments `a`, `b`, and `c`.
You need to create a new directory named `mmflow/engine/optimizers/my_optimizer.py`.
And then implement the new optimizer in a file, e.g., in `mmflow/engine/optimizers/my_optimizer.py`:

```python
from mmflow.registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):

```

#### 2. Add the optimizer to registry

To find the above module defined above, this module should be imported into the main namespace at first. There are two ways to achieve it.

- Modify `mmflow/engine/optimizers/__init__.py` to import it.

  The newly defined module should be imported in `mmflow/engine/optimizers/__init__.py` so that the registry will
  find the new module and add it:

```python
from .my_optimizer import MyOptimizer
```

- Use `custom_imports` in the config to manually import it

`custom_imports` can import module manually as long as the module can be located in `PYTHONPATH`, without modifying source code

```python
custom_imports = dict(imports=['mmflow.engine.optimizers.my_optimizer'], allow_failed_imports=False)
```

The module `mmflow.engine.optimizers.my_optimizer` will be imported at the beginning of the program and the class `MyOptimizer` is then automatically registered.
Note that only the package containing the class `MyOptimizer` should be imported. `mmflow.engine.optimizers.my_optimizer.MyOptimizer` **cannot** be imported directly.

#### 3. Specify the optimizer in the config file

Then you can use `MyOptimizer` in `optimizer` field in `optim_wrapper` field of config files.
In the configs, the optimizers are defined by the field `optimizer` like the following:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
```

To use your own optimizer, the field can be changed to

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
```

### Customize optimizer constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNorm layers.
The users can do those fine-grained parameter tuning through customizing optimizer wrapper constructor.

```python
from mmengine.optim import DefaultOptimWrapperConstructor

from mmflow.registry import OPTIM_WRAPPER_CONSTRUCTORS
from .my_optimizer import MyOptimizer


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimizerConstructor:

    def __init__(self,
                 optimizer_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        pass

    def __call__(self, model: nn.Module) -> OptimWrapper:

        return optim_wrapper
```

The default optimizer wrapper constructor is implemented [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py#L18),
which could also serve as a template for the new optimizer wrapper constructor.

### Additional settings

Tricks not implemented by the optimizer should be implemented through optimizer wrapper constructor (e.g., set parameter-wise learning rates) or hooks.
We list some common settings that could stabilize the training or accelerate the training. Feel free to create PR, issue for more settings.

- **Use gradient clip to stabilize training**:
  Some models need gradient clip to stabilize the training process. An example is as below:

  ```python
    optim_wrapper = dict(
        _delete_=True, clip_grad=dict(max_norm=35, norm_type=2))
  ```

  If your config inherits the base config which already sets the `optim_wrapper`, you might need `_delete_=True` to override the unnecessary settings. See the [config documentation](../user_guides/config.md) for more details.

- **Use momentum schedule to accelerate model convergence**:
  We support momentum scheduler to modify model's momentum according to learning rate, which could make the model converge in a faster way.
  Momentum scheduler is usually used with LR scheduler, for example, the following config is used in 3D detection to accelerate convergence.
  For more details, please refer to the implementation of [CosineAnnealingLR](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L43) and [CosineAnnealingMomentum](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/momentum_scheduler.py#L71).

  ```python
  param_scheduler = [
      # learning rate scheduler
      # During the first 8 epochs, learning rate increases from 0 to lr * 10
      # during the next 12 epochs, learning rate decreases from lr * 10 to lr * 1e-4
      dict(
          type='CosineAnnealingLR',
          T_max=8,
          eta_min=lr * 10,
          begin=0,
          end=8,
          by_epoch=True,
          convert_to_iter_based=True),
      dict(
          type='CosineAnnealingLR',
          T_max=12,
          eta_min=lr * 1e-4,
          begin=8,
          end=20,
          by_epoch=True,
          convert_to_iter_based=True),
      # momentum scheduler
      # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
      # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
      dict(
          type='CosineAnnealingMomentum',
          T_max=8,
          eta_min=0.85 / 0.95,
          begin=0,
          end=8,
          by_epoch=True,
          convert_to_iter_based=True),
      dict(
          type='CosineAnnealingMomentum',
          T_max=12,
          eta_min=1,
          begin=8,
          end=20,
          by_epoch=True,
          convert_to_iter_based=True)
  ]
  ```

## Customize training schedules

By default we use step learning rate with 1x schedule, this calls [MultiStepLR](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py#L139) in MMEngine.
We support many other learning rate schedule [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py), such as `CosineAnnealingLR` and `PolyLR` schedule. Here are some examples

- Poly schedule:

  ```python
  param_scheduler = [
      dict(
          type='PolyLR',
          power=0.9,
          eta_min=1e-4,
          begin=0,
          end=8,
          by_epoch=True)]
  ```

- ConsineAnnealing schedule:

  ```python
  param_scheduler = [
      dict(
          type='CosineAnnealingLR',
          T_max=8,
          eta_min=lr * 1e-5,
          begin=0,
          end=8,
          by_epoch=True)]
  ```

## Customize hooks

### Customize self-implemented hooks

#### 1. Implement a new hook

MMEngine provides many useful [hooks](https://mmengine.readthedocs.io/en/latest/tutorials/hooks.html), but there are some occasions when the users might need to implement a new hook. MMFlow supports customized hooks in training in v1.0. Thus the users could implement a hook directly in mmflow and use the hook by only modifying the config in training.
Here we give an example of creating a new hook in mmflow and using it in training.

```python
from mmengine.hooks import Hook
from mmflow.registry import HOOKS


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):

    def before_run(self, runner) -> None:

    def after_run(self, runner) -> None:

    def before_train(self, runner) -> None:

    def after_train(self, runner) -> None:

    def before_train_epoch(self, runner) -> None:

    def after_train_epoch(self, runner) -> None:

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
```

Depending on the functionality of the hook, the users need to specify what the hook will do at each stage of the training in `before_run`, `after_run`, `before_train`, `after_train` , `before_train_epoch`, `after_train_epoch`, `before_train_iter`, and `after_train_iter`.  There are more points where hooks can be inserted, referring to [base hook class](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/hook.py#L9) for more detail.

#### 2. Register the new hook

Then we need to make `MyHook` imported. Assuming the implementation of `MyHook` is in `mmflow/engine/hooks/my_hook.py`, there are two ways to do that:

- Modify `mmflow/engine/hooks/__init__.py` to import it.

  The newly defined module should be imported in `mmflow/engine/hooks/__init__.py` so that the registry will
  find the new module and add it:

```python
from .my_hook import MyHook
```

- Use `custom_imports` in the config to manually import it

```python
custom_imports = dict(imports=['mmflow.engine.hooks.my_hook'], allow_failed_imports=False)
```

#### 3. Modify the config

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

You can also set the priority of the hook by adding key `priority` to `'NORMAL'` or `'HIGHEST'` as below

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

By default the hook's priority is set as `NORMAL` during registration.

### Use hooks implemented in MMFlow

If the hook is already implemented in MMFlow, you can directly modify the config to use the hook as below.

#### Example: `FlowVisualizationHook`

We implement a customized hook named [FlowVisualizationHook](https://github.com/open-mmlab/mmflow/blob/dev-1.x/mmflow/engine/hooks/visualization_hook.py) to visualize the predicted optical flow. We set it in the schedule config file, such as [schedule_s_long.py](https://github.com/open-mmlab/mmflow/blob/dev-1.x/configs/_base_/schedules/schedule_s_long.py).

### Modify default runtime hooks

There are some common hooks that are not registered through `custom_hooks` but has been registered by default when importing MMEngine, they are

- log_config
- checkpoint_config
- evaluation
- lr_config
- optimizer_config
- momentum_config

In those hooks, only the logger hook has the `VERY_LOW` priority, others' priority are `NORMAL`.
The above-mentioned tutorials already cover how to modify `optimizer_config`, `momentum_config`, and `lr_config`.
Here we reveals how what we can do with `log_config`, `checkpoint_config`, and `evaluation`.

#### Checkpoint config

The MMCV runner will use `checkpoint_config` to initialize [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/hooks/checkpoint.py#L9).

```python
checkpoint_config = dict(interval=1)
```

The users could set `max_keep_ckpts` to only save only small number of checkpoints or decide whether to store state dict of optimizer by `save_optimizer`.
More details of the arguments are [here](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)

#### Log config

The `log_config` wraps multiple logger hooks and enables to set intervals. Now MMCV supports `WandbLoggerHook`, `MlflowLoggerHook`, and `TensorboardLoggerHook`.
The detail usages can be found in the [doc](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook).

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### Evaluation config

The config of `evaluation` will be used to initialize the [`EvalHook`](<>).
Except for the key `interval`, other arguments such as `metric` will be passed to the `online_evaluation()`

```python
evaluation = dict(interval=50000, metric='EPE')
```
