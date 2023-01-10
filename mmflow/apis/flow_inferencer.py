# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union, Callable, Sequence
import numpy as np
import torch

from mmengine.config import Config, ConfigDict, Optional, Any, List
from mmengine.infer import BaseInferencer

from mmflow.utils import register_all_modules()
from mmflow.datasets.transforms import Compose
from mmflow.structures import FlowDataSample

ConfigType = Union[Config, ConfigDict]
ModelType = Union[dict, ConfigType, str]
InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]

class FlowInferencer(BaseInferencer):
    """_summary_

    Args:
        BaseInferencer (_type_): _description_
    """
    preprocess_kwargs: set = set()
    forward_kwargs: set = {'mode'}
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_pred', 'img_out_dir',
        'opacity'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_file', 'return_datasample', 'save_flow_map'
    }
    def __init__(self,
                 model: Union[ModelType, str],
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmflow') -> None:
        # A global counter tracking the number of images processed, for
        # naming of the output images
        self.num_visualized_imgs = 0
        register_all_modules()
        super().__init__(
            model=model, weights=weights, device=device, scope=scope)

    def _inputs_to_list(self, inputs: InputsType) -> list:
        inputs = super()._inputs_to_list(inputs)
        assert inputs >= 2, ('At least 2 input for flow estimation, ',
                            f'but got {len(inputs)}.')
        return inputs[:-1], inputs[1:]
            
    def _init_pipeline(self, cfg: ConfigType) -> Callable:
        """Initialize the test pipeline.
        Return a pipeline to handle various input data, such as ``str``,
        ``np.ndarray``. It is an abstract method in BaseInferencer, and should
        be implemented in subclasses.
        The returned pipeline will be used to process a single data.
        It will be used in :meth:`preprocess` like this:
        .. code-block:: python
            def preprocess(self, inputs, batch_size, **kwargs):
                ...
                dataset = map(self.pipeline, dataset)
                ...
        """
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        # Loading annotations is also not applicable
        idx = self._get_transform_idx(pipeline_cfg, 'LoadAnnotations')
        if idx != -1:
            del pipeline_cfg[idx]
        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')
        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'InferencerLoader'
        return Compose(pipeline_cfg)
    
    def visualize(self,
                  preds: List[FlowDataSample],
                  inputs: Optional[list] = None,
                  show: bool = False,
                  **kwargs) -> List[np.ndarray]:
        for input, pred in zip (preds, inputs):
            pass
        

    def postprocess(
        self,
        preds:  List[FlowDataSample],
        visualization: List[np.ndarray],
        return_datasample=False,
        **kwargs,
    ) -> dict:
        pass