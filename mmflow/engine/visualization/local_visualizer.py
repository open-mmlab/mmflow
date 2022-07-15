# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import mmcv
import numpy as np
from mmengine import Visualizer

from mmflow.data import FlowDataSample
from mmflow.registry import VISUALIZERS


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
                       gt_sample: Optional[FlowDataSample] = None,
                       pred_sample: Optional[FlowDataSample] = None,
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
            image (None): The image to draw. For MMFlow, set to None.
            gt_sample (:obj:`FlowDataSample`, optional): GT FlowDataSample.
                The ground truth of optical flow from img1 to img2,
                which has 3 dimensions in order of  channel, height
                and width, is in the data field of 'gt_flow_fw' in gt_sample.
                Defaults to None.
            pred_sample (:obj:`FlowDataSample`, optional): Prediction
                FlowDataSample. The prediction of optical flow from
                img1 to img2 is in the data field of 'pred_flow_fw'
                in pred_sample. Defaults to None.
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

        if draw_gt and gt_sample is not None:
            assert 'gt_flow_fw' in gt_sample
            gt_flow_fw = gt_sample.gt_flow_fw.data.permute(1, 2, 0).numpy()
            gt_flow_fw_map = np.uint8(mmcv.flow2rgb(gt_flow_fw) * 255.)

        if draw_pred and pred_sample is not None:
            assert 'pred_flow_fw' in pred_sample
            pred_flow_fw = pred_sample.pred_flow_fw.data.permute(
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
