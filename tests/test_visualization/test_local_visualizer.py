# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import PixelData

from mmflow.structures import FlowDataSample
from mmflow.visualization import FlowLocalVisualizer


class TestFlowLocalVisualizer(TestCase):

    def test_add_datasample(self):

        # test gt_flow_fw
        gt_flow_fw = PixelData()
        gt_flow_fw.data = torch.rand((2, 4, 5))
        data_sample = FlowDataSample()
        data_sample.gt_flow_fw = gt_flow_fw

        flow_local_visualizer = FlowLocalVisualizer(
            vis_backends=[dict(type='LocalVisBackend')], save_dir='.')
        flow_local_visualizer.add_datasample('gt_flow_fw', None, data_sample)

        # test gt_flow_fw and pred_flow_fw
        pred_flow_fw = PixelData()
        pred_flow_fw.data = torch.rand(2, 4, 5)
        data_sample.pred_flow_fw = pred_flow_fw

        flow_local_visualizer.add_datasample('gt_pred_flow_fw', None,
                                             data_sample)

        flow_local_visualizer.add_datasample(
            'exclude_gt_flow', None, data_sample, draw_gt=False)

        flow_local_visualizer.add_datasample(
            'exclude_pred_flow', None, data_sample, draw_pred=False)

        flow_local_visualizer.add_datasample(
            'visualization', None, data_sample, show=True, wait_time=1)
