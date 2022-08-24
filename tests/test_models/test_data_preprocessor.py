# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmflow.models import FlowDataPreprocessor
from mmflow.structures import FlowDataSample


class TestFlowDataPreprocessor(TestCase):

    def test_init(self):
        # test mean is None
        processor = FlowDataPreprocessor()
        self.assertTrue(not hasattr(processor, 'mean'))
        self.assertTrue(processor._enable_normalize is False)

        # test mean is not None
        processor = FlowDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])
        self.assertTrue(hasattr(processor, 'mean'))
        self.assertTrue(hasattr(processor, 'std'))
        self.assertTrue(processor._enable_normalize)

        # please specify both mean and std
        with self.assertRaises(AssertionError):
            FlowDataPreprocessor(mean=[0, 0, 0])

        # bgr2rgb and rgb2bgr cannot be set to True at the same time
        with self.assertRaises(AssertionError):
            FlowDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

    def test_forward(self):
        processor = FlowDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        data = [{
            'inputs': torch.randint(0, 256, (2, 3, 11, 10)),
            'data_sample': FlowDataSample()
        }]
        out = processor(data)

        self.assertEqual(out['inputs'].shape, (1, 6, 11, 10))
        self.assertEqual(len(out['data_samples']), 1)

        # test channel_conversion
        processor = FlowDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out = processor(data)
        self.assertEqual(out['inputs'].shape, (1, 6, 11, 10))
        self.assertEqual(len(out['data_samples']), 1)

        # test training and noise
        processor = FlowDataPreprocessor(
            mean=[0., 0., 0.],
            std=[1., 1., 1.],
            sigma_range=(0, 0.04),
            clamp_range=(0., 1.))
        out = processor(data, training=True)
        self.assertEqual(out['inputs'].shape, (1, 6, 11, 10))
        self.assertEqual(len(out['data_samples']), 1)
