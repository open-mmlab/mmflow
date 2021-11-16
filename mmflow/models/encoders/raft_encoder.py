# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule

from ..builder import ENCODERS
from ..utils import BasicBlock, Bottleneck, ResLayer


@ENCODERS.register_module()
class RAFTEncoder(BaseModule):
    """The feature extraction sub-module in RAFT.

    Args:
        in_channels (int): Number of input channels. Defaults to 3.
        out_channels (int): Number of output channels. Defaults to 128.
        net_type (str): The type of this sub-module, if net_type is Basic, the
            residual block is BasicBlock, if net_type is Small, the residual
            block is Bottleneck. Defaults to 'Basic'.
        stem_channels (int, optional): Number of stem channels. If
            stem_channels is None, it will be set based on net_type. If the
            net_type is Basic, the stem_channels is 64, otherwise the
            stem_channels is 32. Defaults to None.
        base_channels (Sequence[int], optional):  Number of base channels of
            res layer. If base_channels is None, it will be set based on
            net_type. If the net_type is Basic, the base_channels is
            (64, 96, 128), otherwise the base_channels is (8, 16, 24).
            Defaults to None.
        num_stages (int, optional): Resnet stages, if it is None, set
            num_stages as length of base_channels. Defaults to None.
        strides (Sequence[int], optional): Strides of the first block of each
            stage. If it is None, it will be (1, 2, 2). Defaults to None.
        dilations (Sequence[int], optional): Dilation of each stage. If it is
            None, it will be (1, 1, 1). Defaults to None.
        deep_stem (bool): Whether Replace 7x7 conv in input stem with 3 3x3
            conv. Defaults to False.
        avg_down (bool): Whether use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None..
        norm_cfg (dict, optional): Config dict for each normalization layer.
            Defaults to dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        plugins (list[dict], optional): List of plugins for stages, each dict
            contains:

            - cfg (dict, required): Cfg dict to build plugin.

            - position (str, required): Position inside block to insert plugin,
            options: 'after_conv1', 'after_conv2', 'after_conv3'.

            - stages (tuple[bool], optional): Stages to apply plugin, length
            should be same as 'num_stages'
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, list, optional): Config of weights initialization.
            Default: None.
    """

    _arch_settings = {
        'Basic': (BasicBlock, (2, 2, 2)),
        'Small': (Bottleneck, (2, 2, 2))
    }

    _stem_channels = {'Basic': 64, 'Small': 32}

    _base_channels = {'Basic': (64, 96, 128), 'Small': (8, 16, 24)}

    _strides = {'Basic': (1, 2, 2), 'Small': (1, 2, 2)}

    _dilations = {'Basic': (1, 1, 1), 'Small': (1, 1, 1)}

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 net_type: str = 'Basic',
                 stem_channels: Optional[int] = None,
                 base_channels: Optional[Sequence[int]] = None,
                 num_stages: Optional[int] = None,
                 strides: Optional[Sequence[int]] = None,
                 dilations: Optional[Sequence[int]] = None,
                 deep_stem: bool = False,
                 avg_down: bool = False,
                 frozen_stages: int = -1,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: dict = dict(type='BN', requires_grad=True),
                 norm_eval: bool = False,
                 plugins: Optional[Sequence[dict]] = None,
                 with_cp: bool = False,
                 init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

        if net_type not in self._arch_settings:
            raise KeyError(f'invalid net type {net_type} for RAFT')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem_channels = (
            stem_channels
            if stem_channels is not None else self._stem_channels[net_type])
        self.base_channels = (
            base_channels
            if base_channels is not None else self._base_channels[net_type])
        self.num_stages = (
            num_stages
            if num_stages is not None else len(self._base_channels[net_type]))
        assert self.num_stages >= 1 and self.num_stages <= 3
        self.strides = (
            strides if strides is not None else self._strides[net_type])
        self.dilations = (
            dilations if dilations is not None else self._dilations[net_type])
        assert len(self.strides) == len(self.dilations) == self.num_stages

        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.plugins = plugins
        self.with_cp = with_cp
        self.block, stage_blocks = self._arch_settings[net_type]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.stem_channels

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None

            planes = self.base_channels[i]
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                plugins=stage_plugins)
            self.inplanes = planes if net_type == 'Basic' else planes * 4
            layer_name = f'res_layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

            last_channel = (
                self.base_channels[-1]
                if net_type == 'Basic' else self.base_channels[-1] * 4)
            self.conv2 = build_conv_layer(
                self.conv_cfg, last_channel, out_channels, kernel_size=1)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=True)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)

    def make_stage_plugins(self, plugins: Sequence[dict],
                           stage_idx: int) -> Sequence[dict]:
        """make plugins for ResNet 'stage_idx'-th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be :
        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:
            conv1-> conv2->conv3->yyy->zzz1->zzz2
        Suppose 'stage_idx=1', the structure of blocks in the stage would be:
            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs) -> torch.nn.Module:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self) -> torch.nn.Module:
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input feature.

        Returns:
            torch.Tensor: Output feature.
        """
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        out = self.conv2(x)
        return out
