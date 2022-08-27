# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import mmcv
import numpy as np
from mmengine.visualization import Visualizer

from mmflow.registry import VISUALIZERS
from mmflow.structures import FlowDataSample


@VISUALIZERS.register_module()
class FlowLocalVisualizer(Visualizer):
    """MMFlow Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Default to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        fig_save_cfg (dict): Keyword parameters of figure for saving.
            Defaults to empty dict.
        fig_show_cfg (dict): Keyword parameters of figure for showing.
            Defaults to empty dict.
    """

    def __init__(self, name='visualizer', **kwargs):
        super().__init__(name, **kwargs)

    def add_datasample(self,
                       name: str,
                       image: Optional[np.ndarray] = None,
                       data_sample: Optional[FlowDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.

        Args:
            name (str): The image identifier.
            image (np.ndarray, optional): The image to draw. For MMFlow,
                set to None.
            data_sample (:obj:`FlowDataSample`, optional): The
                annotation data of every samples. Defaults to None.
            draw_gt (bool): Whether to draw GT FlowDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction FlowDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (int): Delay in milliseconds. 0 is the special
                value that means "forever". Defaults to 0.
            step (int): Global step value to record. Defaults to 0.
        """

        gt_flow_fw_map = None
        pred_flow_fw_map = None

        if draw_gt and data_sample is not None and 'gt_flow_fw' in data_sample:
            gt_flow_fw = data_sample.gt_flow_fw.data.permute(1, 2,
                                                             0).cpu().numpy()
            gt_flow_fw_map = np.uint8(mmcv.flow2rgb(gt_flow_fw) * 255.)

        if (draw_pred and data_sample is not None
                and 'pred_flow_fw' in data_sample):
            pred_flow_fw = data_sample.pred_flow_fw.data.permute(
                1, 2, 0).cpu().numpy()
            pred_flow_fw_map = np.uint8(mmcv.flow2rgb(pred_flow_fw) * 255.)

        if gt_flow_fw_map is not None and pred_flow_fw_map is not None:
            assert gt_flow_fw_map.shape == pred_flow_fw_map.shape
            drawn_img = np.concatenate((gt_flow_fw_map, pred_flow_fw_map),
                                       axis=1)
        elif gt_flow_fw_map is not None:
            drawn_img = gt_flow_fw_map
        else:
            drawn_img = pred_flow_fw_map

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, drawn_img, step)
