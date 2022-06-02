# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch

from mmflow.metrics import EndPointError
from mmflow.metrics.utils import end_point_error_map


def test_end_point_error_map():
    flow_pred = np.random.normal(size=(10, 10, 2))
    flow_gt = np.random.normal(size=(10, 10, 2))

    target = np.linalg.norm((flow_pred - flow_gt), ord=2, axis=-1)
    assert (target == end_point_error_map(flow_pred, flow_gt)).all()


class TestEPEMetric(TestCase):

    flow_pred = np.array([[[10., 5.], [0.1, 3.]], [[3., 15.2], [2.4, 4.5]]])
    flow_gt = np.array([[[10.1, 4.8], [10, 3.]], [[6., 10.2], [2.0, 4.1]]])
    valid_gt = np.array([[1., 1.], [1., 0.3]])

    def _create_toy_data_batch(self):
        flow = torch.from_numpy(self.flow_gt.transpose(2, 0, 1))
        valid = torch.from_numpy(self.valid_gt[None, ...])
        return [dict(data_sample=dict(gt_flow_fw=flow, gt_valid=valid))]

    def _create_toy_predictions(self):
        flow = torch.from_numpy(self.flow_pred.transpose(2, 0, 1))
        return [dict(pred_flow_fw=flow)]

    def test_evaluate(self):
        epe_metric = EndPointError()
        toy_predictions = self._create_toy_predictions()
        toy_data_batch = self._create_toy_data_batch()

        epe_metric.process(toy_data_batch, toy_predictions)
        eval_result = epe_metric.evaluate(1)
        epe_tar = np.linalg.norm((self.flow_pred - self.flow_gt),
                                 ord=2,
                                 axis=-1)
        epe_tar = (epe_tar[0].sum() + epe_tar[1][0]) / 3

        assert eval_result['EPE'] == epe_tar
