# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.infer import BaseInferencer
from rich.progress import track

from mmflow.datasets import write_flow
from mmflow.datasets.transforms import Compose
from mmflow.structures import FlowDataSample
from mmflow.utils import register_all_modules

ConfigType = Union[Config, ConfigDict]
ModelType = Union[dict, ConfigType, str]
InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Sequence[InputType]


class FlowInferencer(BaseInferencer):
    """_summary_

    Args:
        BaseInferencer (_type_): _description_
    """
    preprocess_kwargs: set = set()
    forward_kwargs: set = {'mode'}
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_pred', 'img_out_dir',
        'direction'
    }
    postprocess_kwargs: set = {
        'print_result', 'pred_out_file', 'return_datasample', 'save_flow_map',
        'direction'
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

    def __call__(self,
                 inputs: InputsType,
                 return_datasamples: bool = False,
                 batch_size: int = 1,
                 return_vis: bool = False,
                 show: bool = False,
                 wait_time: int = 0,
                 **kwargs) -> dict:
        """_summary_

        Args:
            inputs (InputsType): _description_
            return_datasamples (bool, optional): _description_. Defaults to False.
            batch_size (int, optional): _description_. Defaults to 1.
            return_vis (bool, optional): _description_. Defaults to False.
            show (bool, optional): _description_. Defaults to False.

            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: _description_
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)
        ori_inputs1, ori_inputs2 = self._inputs_to_list(inputs)
        inputs = self.preprocess(
            ori_inputs1,
            ori_inputs2,
            batch_size=batch_size,
            **preprocess_kwargs)
        preds = []
        for data in track(inputs, description='Inference'):
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(ori_inputs1, preds, **visualize_kwargs)
        results = self.postprocess(preds, visualization, return_datasamples,
                                   **postprocess_kwargs)
        return results

    def _inputs_to_list(self, inputs: InputsType) -> Tuple[list, list]:
        """Preprocess the inputs to a list.

        Split sequence inputs into two list for two adjacent frames:

        - list or tuple: return tuple of list.
        - str:
            - Directory path: return all files in the directory and split two
              list of adjancent frames.
            - Other cases: return a list containing the string and split two
              list of adjancent frames. The string could be a path to file, a
              url or other types of string according to the task.

        Args:
            inputs (InputsType): Inputs for the inferencer.

        Returns:
            Tuple[list]: Tuple of 2 inputs list for the :meth:`preprocess`.
        """
        inputs = super()._inputs_to_list(inputs)
        assert inputs >= 2, ('At least 2 input for flow estimation, ',
                             f'but got {len(inputs)}.')
        return inputs[:-1], inputs[1:]

    def preprocess(self,
                   inputs1: InputsType,
                   inputs2: InputsType,
                   batch_size: int = 1,
                   **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs1 (InputsType): Inputs given by user.
            inputs2 (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(
            map(self.pipeline, inputs1, inputs2), batch_size)
        yield from map(self.collate_fn, chunked_data)

    def visualize(self,
                  inputs: list,
                  preds: List[FlowDataSample],
                  *,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  direction='forward_flow',
                  img_out_dir: str = '') -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (list): Inputs preprocessed by :meth:`_inputs_to_list`.
            preds (list): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            img_out_dir (str): Output directory of images. Defaults to ''.
        """

        if self.visualizer is None or (not show and img_out_dir == ''
                                       and not return_vis):
            return None
        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')
        results = []
        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type:'
                                 f'{type(single_input)}')
            out_file = osp.join(img_out_dir, img_name) if img_out_dir != ''\
                else None

            draw_img = self.visualizer.add_datasample(
                name=img_name,
                data_sample=pred,
                draw_gt=False,
                draw_pred=True,
                show=show,
                direction=direction,
                wait_time=wait_time,
                out_file=out_file)
            results.append(draw_img)
        return results

    def postprocess(
        self,
        preds: List[FlowDataSample],
        visualization: List[np.ndarray],
        return_datasample=False,
        pred_out_dir='',
        save_flow: bool = True,
        direction: str = 'forward',
        **kwargs,
    ) -> dict:

        results_dict = {}

        results_dict['predictions'] = preds
        results_dict['visualization'] = visualization
        flow_direction = 'pred_flow_fw' if direction == 'forward' \
            else 'pred_flow_bw'

        if pred_out_dir != '':
            mmengine.mkdir_or_exist(pred_out_dir)
            if save_flow:
                for i, pred in enumerate(preds):
                    pred_num = str(i).zfill(8)
                    flow_name = f'{pred_num}.jpg'
                    out_file = osp.join(pred_out_dir, flow_name)
                    write_flow(pred[flow_direction].data, out_file)

        if return_datasample:
            return preds

        return results_dict

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

    def _get_transform_idx(self, pipeline_cfg: ConfigType, name: str) -> int:
        """Returns the index of the transform in a pipeline.
        If the transform is not found, returns -1.
        """
        for i, transform in enumerate(pipeline_cfg):
            if transform['type'] == name:
                return i
        return -1
