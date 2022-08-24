# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import numpy as np
import torch
from mmengine.data import PixelData

from mmflow.datasets.transforms import PackFlowInputs
from mmflow.structures import FlowDataSample

img1_path = osp.join(osp.dirname(__file__), '../data/00001_img1.ppm')
img2_path = osp.join(osp.dirname(__file__), '../data/00001_img2.ppm')
flow_path = osp.join(osp.dirname(__file__), '../data/00001_flow.flo')
flow_fw_path = osp.join(osp.dirname(__file__), '../data/0000000-flow_01.flo')
flow_bw_path = osp.join(osp.dirname(__file__), '../data/0000000-flow_10.flo')
occ_fw_path = osp.join(osp.dirname(__file__), '../data/0000000-occ_01.png')
occ_bw_path = osp.join(osp.dirname(__file__), '../data/0000000-occ_10.png')
sparse_flow_fw_path = osp.join(
    osp.dirname(__file__), '../data/sparse_flow.png')


class TestPackFlowInputs(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.result1 = {
            'img1_path': img1_path,
            'img2_path': img2_path,
            'img1': rng.rand(30, 40, 3),
            'img2': rng.rand(30, 40, 3),
            'flow_fw_path': flow_path,
            'gt_flow_fw': rng.rand(30, 40, 2),
            'gt_flow_bw': None,
            'gt_occ_fw': None,
            'gt_occ_fw': None,
            'ori_shape': (30, 40),
            'img_shape': (30, 40),
            'scale_factor': 1.0
        }
        self.result2 = {
            'img1_path': img1_path,
            'img2_path': img2_path,
            'img1': rng.rand(30, 40, 3),
            'img2': rng.rand(30, 40, 3),
            'flow_fw_path': flow_path,
            'gt_flow_fw': rng.rand(30, 40, 2),
            'gt_valid_fw': rng.rand(30, 40),
            'gt_valid_bw': rng.rand(30, 40),
            'gt_flow_bw': None,
            'gt_occ_fw': None,
            'gt_occ_fw': None,
            'ori_shape': (30, 40),
            'img_shape': (30, 40),
            'scale_factor': 1.0
        }
        self.result3 = {
            'img1_path': img1_path,
            'img2_path': img2_path,
            'img1': rng.rand(30, 40, 3),
            'img2': rng.rand(30, 40, 3),
            'flow_fw_path': flow_path,
            'gt_flow_fw': rng.rand(30, 40, 2),
            'gt_flow_bw': None,
            'gt_occ_fw': rng.rand(30, 40),
            'gt_occ_bw': None,
            'ori_shape': (30, 40),
            'img_shape': (30, 40),
            'scale_factor': 1.0
        }
        self.meta_keys = ('img1_path', 'img2_path', 'ori_shape', 'img_shape',
                          'scale_factor', 'flip')

    def test_transform(self):
        transform = PackFlowInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.result1))

        self.assertIn('data_samples', results)
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['inputs'].shape, (2, 3, 30, 40))
        self.assertIsInstance(results['data_samples'], FlowDataSample)
        self.assertIsInstance(results['data_samples'].gt_flow_fw, PixelData)
        self.assertIsInstance(results['data_samples'].gt_flow_fw.data,
                              torch.Tensor)
        self.assertEqual(results['data_samples'].gt_flow_fw.shape, (30, 40))
        self.assertEqual(results['data_samples'].metainfo['scale_factor'], 1.)
        self.assertEqual(results['data_samples'].metainfo['ori_shape'],
                         (30, 40))
        self.assertEqual(results['data_samples'].metainfo['img_shape'],
                         (30, 40))

    def test_sparse_flow_transform(self):
        transform = PackFlowInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.result2))

        self.assertIn('data_samples', results)
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['inputs'].shape, (2, 3, 30, 40))
        self.assertIsInstance(results['data_samples'], FlowDataSample)
        self.assertIsInstance(results['data_samples'].gt_flow_fw, PixelData)
        self.assertIsInstance(results['data_samples'].gt_flow_fw.data,
                              torch.Tensor)
        self.assertEqual(results['data_samples'].gt_flow_fw.shape, (30, 40))
        self.assertIsInstance(results['data_samples'].gt_valid_fw, PixelData)
        self.assertIsInstance(results['data_samples'].gt_valid_fw.data,
                              torch.Tensor)
        self.assertEqual(results['data_samples'].gt_valid_fw.shape, (30, 40))

        self.assertEqual(results['data_samples'].metainfo['scale_factor'], 1.)
        self.assertEqual(results['data_samples'].metainfo['ori_shape'],
                         (30, 40))
        self.assertEqual(results['data_samples'].metainfo['img_shape'],
                         (30, 40))

    def test_occ_flow_transform(self):
        transform = PackFlowInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.result3))

        self.assertIn('data_samples', results)
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertEqual(results['inputs'].shape, (2, 3, 30, 40))
        self.assertIsInstance(results['data_samples'], FlowDataSample)
        self.assertIsInstance(results['data_samples'].gt_flow_fw, PixelData)
        self.assertIsInstance(results['data_samples'].gt_flow_fw.data,
                              torch.Tensor)
        self.assertEqual(results['data_samples'].gt_flow_fw.shape, (30, 40))
        self.assertIsInstance(results['data_samples'].gt_occ_fw, PixelData)
        self.assertIsInstance(results['data_samples'].gt_occ_fw.data,
                              torch.Tensor)
        self.assertEqual(results['data_samples'].gt_occ_fw.shape, (30, 40))

        self.assertEqual(results['data_samples'].metainfo['scale_factor'], 1.)
        self.assertEqual(results['data_samples'].metainfo['ori_shape'],
                         (30, 40))
        self.assertEqual(results['data_samples'].metainfo['img_shape'],
                         (30, 40))

    def test_repr(self):
        transform = PackFlowInputs(meta_keys=self.meta_keys)

        self.assertEqual(
            repr(transform), f'PackFlowInputs(meta_keys={self.meta_keys})')
