# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from mmflow.ops import build_operators


class LinkOutput:
    """The link output between two estimators in FlowNet2."""

    def __init__(self) -> None:
        self.upsample_flow = None
        self.scaled_flow = None
        self.norm_scaled_flow = None
        self.warped_img2 = None
        self.diff = None
        self.brightness_err = None


class BasicLink(nn.Module):
    """Connect two separate flow estimators.

    BasicLink compute the following 5 values: upsampled flow prediction,
    normalized upsampled flow prediction, warped image, difference between the
    first image and warped image, brightness error.

    Args:
        scale_factor (int): Scale factor of upsampling. Default to 4.
        mode (str): Algorithm used for upsampling: 'nearest' , 'linear' ,
            'bilinear' , 'bicubic' , 'trilinear' , 'area'. Default: 'bilinear'.
        warp_cfg (dict): Config for warp operator. Default to
            dict(type='Warp', padding_mode='border', align_corners=True))
    """

    def __init__(self,
                 scale_factor: int = 4,
                 mode: str = 'bilinear',
                 warp_cfg: dict = dict(
                     type='Warp', padding_mode='border', align_corners=True)):
        super(BasicLink, self).__init__()

        self.warp = build_operators(warp_cfg)
        self.upSample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def __call__(self, img1: Tensor, img2: Tensor, flow: Tensor,
                 flow_div: float) -> LinkOutput:
        """Call function for BasicLink.

        Args:
            img1 (Tensor): The first input image.
            img2 (Tensor): The second input images.
            flow (Tensor): The estimated optical flow from the first image to
                the second image.
            flow_div (float): The divisor for scaling the value of optical
                flow.

        Returns:
            LinkOutput: The output for the next flow estimator.
        """
        upsample_flow = self.upSample(flow)
        scaled_flow = self.upSample(flow) * flow_div
        norm_scaled_flow = torch.norm(scaled_flow, p=2, dim=1, keepdim=True)
        warped_img2 = self.warp(img2, scaled_flow)
        diff = img1 - warped_img2
        bright_err = torch.norm(diff, p=2, dim=1, keepdim=True)

        output = LinkOutput()
        output.upsample_flow = upsample_flow
        output.scaled_flow = scaled_flow
        output.norm_scaled_flow = norm_scaled_flow
        output.warped_img2 = warped_img2
        output.diff = diff
        output.brightness_err = bright_err

        return output
