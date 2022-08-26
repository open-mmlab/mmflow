# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmflow.registry import METRICS
from .utils import end_point_error_map


@METRICS.register_module()
class FlowOutliers(BaseMetric):
    """Optical flow outliers metric.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        gt_flow_list = []
        pred_flow_list = []
        gt_valid_list = []

        for data_sample in data_samples:
            # sintel and kitti only support forward optical flow
            # so here we only evaluate the predicted forward flow
            # tensor with shape (2, H, W) to ndarray with shape (H, W, 2)
            gt_flow = data_sample['gt_flow_fw']['data'].permute(
                1, 2, 0).cpu().numpy()
            pred_flow = data_sample['pred_flow_fw']['data'].permute(
                1, 2, 0).cpu().numpy()

            if data_sample.get('gt_valid_fw', None) is not None:
                # tensor with shape (1, H, W) to ndarray with shape (H, W)
                gt_valid = np.squeeze(
                    data_sample['gt_valid_fw']['data'].cpu().numpy().squeeze())
            else:
                gt_valid = np.ones_like(gt_flow[..., 0])
            gt_flow_list.append(gt_flow)
            pred_flow_list.append(pred_flow)
            gt_valid_list.append(gt_valid)
        fl = self.optical_flow_outliers(pred_flow_list, gt_flow_list,
                                        gt_valid_list)
        self.results.append(fl)

    @staticmethod
    def optical_flow_outliers(flow_pred: Sequence[np.ndarray],
                              flow_gt: Sequence[np.ndarray],
                              valid_gt: Sequence[np.ndarray]) -> float:
        """Calculate percentage of optical flow outliers for KITTI dataset.

        Args:
            flow_pred (list): output list of flow map from flow_estimator
                shape(H, W, 2).
            flow_gt (list): ground truth list of flow map shape(H, W, 2).
            valid_gt (list): the list of valid mask for ground truth with the
                shape (H, W).

        Returns:
            float: optical flow outliers for output.
        """
        out_list = []
        assert len(flow_pred) == len(flow_gt) == len(valid_gt)
        for _flow_pred, _flow_gt, _valid_gt in zip(flow_pred, flow_gt,
                                                   valid_gt):
            epe_map = end_point_error_map(_flow_pred, _flow_gt)
            epe = epe_map.reshape(-1)
            mag_map = np.sqrt(np.sum(_flow_gt**2, axis=-1))
            mag = mag_map.reshape(-1) + 1e-6
            val = _valid_gt.reshape(-1) >= 0.5
            # 3.0 and 0.05 is token from KITTI devkit
            # Inliers are defined as EPE < 3 pixels or < 5%
            out = ((epe > 3.0) & ((epe / mag) > 0.05)).astype(float)
            out_list.append(out[val])
        out_list = np.concatenate(out_list)
        fl = 100 * np.mean(out_list)

        return fl

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        eval_result = dict(Fl=np.array(results).mean().item())
        return eval_result
