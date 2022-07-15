# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import PixelData

from mmflow.data import FlowDataSample
from mmflow.engine.visualization import FlowLocalVisualizer


class TestFlowLocalVisualizer(TestCase):

    def test_add_datasample(self):

        # test gt_flow_fw
        gt_flow = torch.rand((2, 4, 5))
        gt_flow_fw = PixelData()
        gt_flow_fw.data = gt_flow
        gt_data_sample = FlowDataSample()
        gt_data_sample.gt_flow_fw = gt_flow_fw

        flow_local_visualizer = FlowLocalVisualizer(
            vis_backends=[dict(type='LocalVisBackend')], save_dir='.')
        flow_local_visualizer.add_datasample('gt_flow_fw', None,
                                             gt_data_sample)

        # test gt_flow_fw and pred_flow_fw
        pred_flow = torch.rand(2, 4, 5)
        pred_flow_fw = PixelData()
        pred_flow_fw.data = pred_flow
        pred_data_sample = FlowDataSample()
        pred_data_sample.pred_flow_fw = pred_flow_fw

        flow_local_visualizer.add_datasample('gt_pred_flow_fw', None,
                                             gt_data_sample, pred_data_sample)

        flow_local_visualizer.add_datasample(
            'exclude_gt_flow',
            None,
            gt_data_sample,
            pred_data_sample,
            draw_gt=False)

        flow_local_visualizer.add_datasample(
            'exclude_pred_flow',
            None,
            gt_data_sample,
            pred_data_sample,
            draw_pred=False)

        flow_local_visualizer.add_datasample(
            'visualization',
            None,
            gt_data_sample,
            pred_data_sample,
            show=True,
            wait_time=1)
