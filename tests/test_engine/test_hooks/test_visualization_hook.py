# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.data import PixelData

from mmflow.datasets import read_flow
from mmflow.engine.hooks import FlowVisualizationHook
from mmflow.structures import FlowDataSample
from mmflow.visualization import FlowLocalVisualizer


def generate_data_sample(img1_path, img2_path, flow_path) -> FlowDataSample:
    data_sample = FlowDataSample()
    data_sample.set_metainfo({'img1_path': img1_path, 'img2_path': img2_path})
    gt_flow = read_flow(flow_path)
    gt_flow_fw = PixelData()
    gt_flow_fw.data = torch.from_numpy(gt_flow).permute(2, 0, 1)
    data_sample.gt_flow_fw = gt_flow_fw
    return data_sample


class TestFlowVisualizationHook(TestCase):

    def setUp(self) -> None:
        if not FlowLocalVisualizer.check_instance_created('visualizer'):
            FlowLocalVisualizer.get_instance(
                'visualizer',
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir='.')

        data_sample1 = generate_data_sample(
            img1_path=osp.join(
                osp.dirname(__file__), '../../data/0000000-img_0.png'),
            img2_path=osp.join(
                osp.dirname(__file__), '../../data/0000000-img_1.png'),
            flow_path=osp.join(
                osp.dirname(__file__), '../../data/0000000-flow_01.flo'))

        data_sample2 = generate_data_sample(
            img1_path=osp.join(
                osp.dirname(__file__), '../../data/0000000-img_1.png'),
            img2_path=osp.join(
                osp.dirname(__file__), '../../data/0000000-img_0.png'),
            flow_path=osp.join(
                osp.dirname(__file__), '../../data/0000000-flow_10.flo'))
        self.data_batch = [{
            'data_sample': data_sample1
        }, {
            'data_sample': data_sample2
        }]

        pred_flow_fw = PixelData()
        pred_flow_fw.data = torch.rand((2, 50, 50))
        data_sample1.pred_flow_fw = pred_flow_fw

        pred_flow_fw = PixelData()
        pred_flow_fw.data = torch.rand((2, 50, 50))
        data_sample2.pred_flow_fw = pred_flow_fw
        self.outputs = [data_sample1, data_sample2]

    def test_after_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = FlowVisualizationHook(draw=True, interval=1)
        hook._after_iter(
            runner, 1, self.data_batch, self.outputs, mode='train')
        hook._after_iter(runner, 1, self.data_batch, self.outputs, mode='val')
        hook._after_iter(runner, 1, self.data_batch, self.outputs, mode='test')

    def test_after_val_iter(self):
        runner = Mock()
        runner.iter = 2
        hook = FlowVisualizationHook(interval=1)
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

        hook = FlowVisualizationHook(draw=True, interval=1)
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

        hook = FlowVisualizationHook(
            draw=True, interval=1, show=True, wait_time=1)
        hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        runner.iter = 3
        hook = FlowVisualizationHook(draw=True, interval=1)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)
