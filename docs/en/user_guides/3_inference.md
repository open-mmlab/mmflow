# Tutorial 3: Inference with existing models

MMFlow provides pre-trained models for flow estimation in [Model Zoo](../model_zoo.md), and supports multiple standard datasets, including FlyingChairs, Sintel, etc.
This note will show how to use existing models to inference on given images.
As for how to test existing models on standard datasets, please see this [guide](https://github.com/open-mmlab/mmflow/blob/dev-1.x/docs/en/user_guides/4_train_test.md#Test-models-on-standard-datasets)

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

An image demo can be found in [demo/image_demo.py](https://github.com/open-mmlab/mmflow/blob/dev-1.x/demo/image_demo.py).
