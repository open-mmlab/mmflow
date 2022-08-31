# Adding New Modules

MMFlow decomposes a flow estimation method `flow_estimator` into `encoder` and `decoder`. This tutorial is for how to add new components.

## Add a new encoder

1. Create a new file `mmflow/models/encoders/my_encoder.py`.

   You can write a new head inherit from `BaseModule` from mmengine, and overwrite `forward`.
   We have a unified interface for weight initialization in mmengine,
   you can use `init_cfg` to specify the initialization function and arguments,
   or overwrite `init_weights` if you prefer customized initialization.

   ```python
   from mmengine.model import BaseModule

   from mmflow.registry import MODELS

   @MODELS.register_module()
   class MyEncoder(BaseModule):

       def __init__(self, arg1, arg2):  # arg1 and arg2 need to be specified in config
           pass

       def forward(self, x):  # should return a dict
           pass

       # optional
       def init_weights(self):
           pass
   ```

2. Import the module in `mmflow/models/encoders/__init__.py`.

   ```python
   from .my_model import MyEncoder
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

       def __init__(self, arg1, arg2):  # arg1 and arg2 need to be specified in config
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

   `batch_data_samples` contains the ground truth and `batch_img_metas` contains the information of original input images, such as original shape.
   `loss_by_feat` is the loss function to compute the losses between the model output and target,
   and you can refer to the implementation of [PWCNetDecoder](https://github.com/open-mmlab/mmflow/blob/dev-1.x/mmflow/models/decoders/pwcnet_decoder.py).
   `predict_by_feat` aims to restore the flow shape as the original shape of input images,
   and you can refer to the implementations of [BaseDecoder](https://github.com/open-mmlab/mmflow/blob/dev-1.x/mmflow/models/decoders/base_decoder.py)

2. Import the module in `mmflow/models/decoders/__init__.py`

   ```python
   from .my_decoder import MyDecoder
   ```

## Add a new flow_estimator

1. Create a new file `mmflow/models/flow_estimators/my_estimator.py`

   You can write a new flow estimator inherit from `FlowEstimator` like PWC-Net.
   A typical encoder-decoder estimator can be written like:

   ```python
   from .base_flow_estimator import FlowEstimator

   from mmflow.registry import MODELS

   @MODELS.register_module()
   class MyEstimator(FlowEstimator):

       def __init__(self, encoder: dict, decoder: dict):
           pass

       def loss(self, batch_inputs, batch_data_samples):
           pass

       def predict(self, batch_inputs, batch_data_samples):
           pass

       def _forward(self, batch_inputs, data_samples):
           pass

       def extract_feat(self, batch_inputs):
           pass
   ```

   `loss`, `predict`, `_forward` and `extract_feat` are abstract methods of `FlowEstimator`.
   They can be seen as high-level APIs of the methods in `MyEncoder` and `MyDecoder`.

2. Import the module in `mmflow/models/flow_estimators/__init__.py`

   ```python
   from .my_estimator import MyEstimator
   ```

3. Use it in your config file.

   It's worth pointing out that `data_preprocessor` is an important parameter of `FlowEstimator`
   which can be used to move data to a specified device (such as a GPU) and further format the input data.
   In addition, image normalization, adding Gaussian noise are implemented in `data_preprocessor` as well.
   Therefore, `data_preprocessor` needs to be specified in the config of `MyEstimator`.
   You can refer to the config of [PWC-Net](https://github.com/open-mmlab/mmflow/blob/dev-1.x/configs/_base_/models/pwcnet.py) for a typical configuration of `data_preprocessor`.

   ```python
   model = dict(
       type='MyEstimator',
       data_preprocessor=dict(
           type='FlowDataPreprocessor',
           mean=[0., 0., 0.],
           std=[255., 255., 255.]),
       encoder=dict(
           type='MyEncoder',
           arg1=xxx,
           arg2=xxx),
       decoder=dict(
           type='MyDecoder',
           arg1=xxx,
           arg2=xxx))
   ```

## Add new loss

1. Create a new file `mmflow/models/losses/my_loss.py`

   Assume you want to add a new loss as `MyLoss` for flow estimation.

   ```python
   import torch.nn as nn

   from mmflow.registry import MODELS

   def my_loss(pred, target, *args):
       pass

   @MODELS.register_module()
   class MyLoss(nn.Module):

       def __init__(self, *args):
           super(MyLoss, self).__init__()

       def forward(self, preds_dict, target, *args):
           return my_loss(preds_dict, target, *args)
   ```

2. Import the module in `mmflow/models/losses/__init__.py`.

   ```python
   from .my_loss import MyLoss, my_loss
   ```

3. Modify the `flow_loss` field in the model to use `MyLoss`

   ```python
   flow_loss=dict(type='MyLoss')
   ```
