# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmflow.models.utils import BasicLink


def test_basiclink():
    link = BasicLink(scale_factor=2)

    img1 = torch.randn((1, 3, 6, 6))
    img2 = torch.randn((1, 3, 6, 6))

    flow = torch.zeros((1, 2, 3, 3))
    flow_div = 1.

    output = link(img1=img1, img2=img2, flow=flow, flow_div=flow_div)

    assert torch.all(output.upsample_flow == torch.zeros((1, 2, 6, 6)))
    assert torch.all(output.scaled_flow == torch.zeros((1, 2, 6, 6)))
    assert torch.all(output.norm_scaled_flow == torch.zeros((1, 2, 6, 6)))
    assert torch.allclose(output.warped_img2, img2, rtol=1e-5, atol=1e-5)
    assert torch.allclose(output.diff, img1 - img2, rtol=1e-5, atol=1e-5)
    assert torch.allclose(
        output.brightness_err,
        torch.norm(img1 - img2, p=2, dim=1, keepdim=True),
        rtol=1e-5,
        atol=1e-5)
