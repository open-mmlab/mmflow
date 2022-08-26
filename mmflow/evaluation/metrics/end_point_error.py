# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmflow.registry import METRICS
from .utils import end_point_error_map


@METRICS.register_module()
class EndPointError(BaseMetric):
    """End point error metric.

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
        epe = self._end_point_error(
            pred_flow=pred_flow_list,
            gt_flow=gt_flow_list,
            gt_valid=gt_valid_list)
        self.results.append(epe)

    @staticmethod
    def _end_point_error(pred_flow: Sequence[np.ndarray],
                         gt_flow: Sequence[np.ndarray],
                         gt_valid: Sequence[np.ndarray]) -> float:
        """Calculate end point errors between prediction and ground truth.

        Args:
            pred_flow (list): output list of flow map from flow_estimator
                shape(H, W, 2).
            gt_flow (list): ground truth list of flow map shape(H, W, 2).
            gt_valid (list): the list of valid mask for ground truth with the
                shape (H, W).

        Returns:
            float: end point error for output.
        """
        epe_list = []
        assert len(pred_flow) == len(gt_flow)
        for _pred_flow, _gt_flow, _gt_valid in zip(pred_flow, gt_flow,
                                                   gt_valid):
            epe_map = end_point_error_map(_pred_flow, _gt_flow)
            val = _gt_valid.reshape(-1) >= 0.5
            epe_list.append(epe_map.reshape(-1)[val])

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)

        return epe

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        eval_result = dict(EPE=np.array(results).mean().item())
        return eval_result
