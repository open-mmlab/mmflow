from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmflow.ops.builder import build_operators
from ..builder import DECODERS
from .base_decoder import BaseDecoder


class BasicLayers(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels=(32, 64, 32, 16, 2),
                 kernel_size=7,
                 stride=1,
                 act_cfg=dict(type='ReLU', inplace=False),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        convs = []
        in_ch = in_channels
        for out_ch in out_channels[:-1]:
            convs.append(
                ConvModule(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    act_cfg=act_cfg))
            in_ch = out_ch
        convs.append(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_channels[-1],
                kernel_size=kernel_size,
                padding=kernel_size // 2))
        self.layers = nn.Sequential(*convs)

    def forward(self, x):
        return self.layers(x)


@DECODERS.register_module()
class SpyNetDecoder(BaseDecoder):

    def __init__(self,
                 in_channels,
                 pyramid_levels,
                 out_channels=(32, 64, 32, 16, 2),
                 kernel_size=7,
                 stride=1,
                 warp_cfg: dict = dict(type='Warp', align_corners=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        self.pyramid_levels.sort()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.act_cfg = act_cfg

        self.warp = build_operators(warp_cfg)

        layers = []

        for level in self.pyramid_levels:

            layers.append([level, self.make_layers()])

        self.decoders = nn.ModuleDict(layers)

    def make_layers(self):
        return BasicLayers(
            in_channels=self.in_channels, out_channels=self.out_channels)

    def forward(self, imgs1, imgs2):
        flow = None

        residual_flow_preds = dict()
        previous_flow_preds = dict()
        for level in self.pyramid_levels[::-1]:

            img1 = imgs1[level]
            img2 = imgs2[level]
            _, _, H, W = img1.shape

            if flow is None:
                flow = torch.zeros(1, 2, H, W).to(img1)
            else:
                flow = F.interpolate(
                    flow, scale_factor=2, mode='bilinear',
                    align_corners=False) * 2.0

            warped_img2 = self.warp(img2, flow)
            previous_flow_preds[level] = flow

            in_feat = torch.cat((img1, warped_img2, flow), dim=1)

            residual_flow = self.decoders[level](in_feat)
            flow += residual_flow

            residual_flow_preds[level] = residual_flow

        return flow, residual_flow_preds, previous_flow_preds

    def losses(
            self,
            residual_flow_preds: Dict[str, torch.Tensor],
            previous_flow_preds: Dict[str, torch.Tensor],
            flow_gt: torch.Tensor,
            valid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute optical flow loss.

        Args:
            flow_pred (Dict[str, Tensor]): multi-level predicted optical flow.
            flow_gt (Tensor): The ground truth of optical flow.
            valid (Tensor, optional): The valid mask. Defaults to None.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        loss = dict()
        loss['loss_flow'] = self.flow_loss(residual_flow_preds,
                                           previous_flow_preds, flow_gt, valid)
        return loss

    def forward_train(self, imgs1, imgs2, flow_gt, valid=None):
        _, residual_flow_preds, previous_flow_preds = self.forward(
            imgs1=imgs1, imgs2=imgs2)

        return self.losses(
            residual_flow_preds=residual_flow_preds,
            previous_flow_preds=previous_flow_preds,
            flow_gt=flow_gt,
            valid=valid)

    def forward_test(self, imgs1, imgs2, img_metas=None):
        flow, _, _ = self.forward(imgs1=imgs1, imgs2=imgs2)
        flow_result = flow.permute(0, 2, 3, 1).cpu().data.numpy()

        # unravel batch dim,
        flow_result = list(flow_result)
        flow_result = [dict(flow=f) for f in flow_result]

        return self.get_flow(flow_result, img_metas=img_metas)
