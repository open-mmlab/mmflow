# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock, patch

import torch.nn as nn
from torch.testing import assert_allclose

from mmflow.engine.hooks import LiteFlowNetStageLoadHook


def generate_decoders():
    layers = []

    conv_level6_1 = nn.Conv2d(
        in_channels=3, out_channels=5, kernel_size=3, bias=False)
    conv_level6_1.weight.data.zero_()
    conv_level6_2 = nn.Conv2d(
        in_channels=5, out_channels=7, kernel_size=3, bias=False)
    conv_level6_2.weight.data.zero_()
    conv_level6 = nn.Sequential(conv_level6_1, conv_level6_2)
    layers.append(['level6', conv_level6])

    conv_level5_1 = nn.Conv2d(
        in_channels=3, out_channels=5, kernel_size=3, bias=False)
    conv_level5_2 = nn.Conv2d(
        in_channels=5, out_channels=10, kernel_size=3, bias=False)
    conv_level5 = nn.Sequential(conv_level5_1, conv_level5_2)
    layers.append(['level5', conv_level5])

    decoders = nn.ModuleDict(layers)
    return decoders


class TestLiteFlowNetStageLoadHook(TestCase):

    @patch('mmflow.engine.hooks.liteflownet_stage_loading.is_model_wrapper')
    def test_is_model_wrapper_and_before_run(self, mock_is_model_wrapper):
        mock_is_model_wrapper.return_value = True
        runner = Mock()
        runner.model = Mock()
        runner.model.module = Mock()
        runner.model.module.decoder = Mock()
        runner.model.module.decoder.decoders = generate_decoders()

        hook = LiteFlowNetStageLoadHook('level6', 'level5')
        hook.before_run(runner)
        assert_allclose(
            runner.model.module.decoder.decoders['level6'][0].weight.data,
            runner.model.module.decoder.decoders['level5'][0].weight.data)
        return

    def test_is_not_model_wrapper_and_before_run(self):
        runner = Mock()
        runner.model = Mock()
        runner.model = Mock()
        runner.model.decoder = Mock()
        runner.model.decoder.decoders = generate_decoders()

        hook = LiteFlowNetStageLoadHook('level6', 'level5')
        hook.before_run(runner)
        assert_allclose(runner.model.decoder.decoders['level6'][0].weight.data,
                        runner.model.decoder.decoders['level5'][0].weight.data)
        return
