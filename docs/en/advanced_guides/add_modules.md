# Adding New Modules

MMFlow decomposes a flow estimation method `flow_estimator` into `encoder` and `decoder`. This tutorial is for how to add new components.

## Add a new encoder

1. Create a new file `mmflow/models/encoders/my_model.py`.

You can write a new head inherit from `BaseModule` from mmengine, and overwrite `forward`.
We have a unified interface for weight initialization in mmengine,
you can use `init_cfg` to specify the initialization function and arguments,
or overwrite `init_weights` if you prefer customized initialization.

```python
from mmengine.model import BaseModule

from mmflow.registry import MODELS

@MODELS.register_module()
class MyModel(BaseModule):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a dict
        pass

    # optional
    def init_weights(self):
        pass
```

2. Import the module in `mmflow/models/encoders/__init__.py`.

```python
from .my_model import MyModel
```

## Add a new decoder

1. Create a new file `mmflow/models/decoders/my_decoder.py`.

You can write a new head inherit from `BaseModule` from mmengine,
and overwrite `forward` and `init_weights`.

```python
from mmengine.model import BaseModule

from mmflow.registry import MODELS

@MODELS.register_module()
class MyDecoder(BaseModule):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, *args):
        pass

    # optional
    def init_weights(self):
        pass

    def loss(self, *args, batch_data_samples):
        flow_pred = self.forward(*args)
        return self.loss_by_feat(flow_pred, batch_data_samples)

    def predict(self, *args, batch_img_metas):
        flow_pred = self.forward(*args)
        flow_results = flow_pred[self.end_level]
        return self.predict_by_feat(flow_results, batch_img_metas)
```

`loss_by_feat` is the loss function to compute the losses between the model output and target.
`predict_by_feat` is implemented in `BaseDecoder` to restore the flow shape as the original shape of input images.

2. Import the module in `mmflow/models/decoders/__init__.py`

```python
from .my_decoder import MyDecoder
```

## Add a new flow_estimator

1. Create a new file `mmflow/models/flow_estimators/my_estimator.py`

You can write a new flow estimator inherit from `FlowEstimator` like PWC-Net, and implement `forward_train` and `forward_test`

```python
from ..builder import FLOW_ESTIMATORS
from .base import FlowEstimator


@MODELS.register_module()
class MyEstimator(FlowEstimator):

    def __init__(self, arg1, arg2):
        pass

    def forward_train(self, imgs):
        pass

    def forward_test(self, imgs):
        pass
```

2. Import the module in `mmflow/models/flow_estimator/__init__.py`

```python
from .my_estimator import MyEstimator
```

3. Use it in your config file.

we set the module type as `MyEstimator`.

```python
model = dict(
    type='MyEstimator',
    encoder=dict(
        type='MyModel',
        arg1=xxx,
        arg2=xxx),
    decoder=dict(
        type='MyDecoder',
        arg1=xxx,
        arg2=xxx))
```

## Add new loss

Assume you want to add a new loss as `MyLoss`, for flow estimation.
To add a new loss function, the users need implement it in `mmflow/models/losses/my_loss.py`.

```python
import torch
import torch.nn as nn

from mmflow.models import LOSSES

def my_loss(pred, target):
    pass

@MODELS.register_module()
class MyLoss(nn.Module):

    def __init__(self, arg1):
        super(MyLoss, self).__init__()


    def forward(self, output, target):
        return my_loss(output, target)
```

Then the users need to add it in the `mmflow/models/losses/__init__.py`.

```python
from .my_loss import MyLoss, my_loss

```

To use it, modify the `flow_loss` field in the model.

```python
flow_loss=dict(type='MyLoss', use_target_weight=False)
```
