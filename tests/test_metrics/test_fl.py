# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch

from mmflow.metrics import FlowOutliers


class TestFlMetric(TestCase):

    flow_pred = np.array([[[10., 5.], [0.1, 3.]], [[3., 15.2], [2.4, 4.5]]])
    flow_gt = np.array([[[10.1, 4.8], [10, 3.]], [[6., 10.2], [2.0, 4.1]]])
    valid_gt = np.array([[1., 1.], [1., 0.3]])

    def _create_toy_data_batch(self):
        flow = torch.from_numpy(self.flow_gt.transpose(2, 0, 1))
        valid = torch.from_numpy(self.valid_gt[None, ...])
        return [
            dict(
                data_sample=dict(
                    gt_flow_fw=dict(data=flow), gt_valid=dict(data=valid)))
        ]

    def _create_toy_predictions(self):
        flow = torch.from_numpy(self.flow_pred.transpose(2, 0, 1))
        return [dict(pred_flow_fw=dict(data=flow))]

    def test_evaluate(self):
        fl_metric = FlowOutliers()
        toy_predictions = self._create_toy_predictions()
        toy_data_batch = self._create_toy_data_batch()

        fl_metric.process(toy_data_batch, toy_predictions)
        eval_result = fl_metric.evaluate(1)
        fl_tar = 100 * (2 / 3)
        assert eval_result['Fl'] == fl_tar
