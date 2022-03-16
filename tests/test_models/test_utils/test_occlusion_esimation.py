# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.utils import occlusion_estimation


def test_occlusion_estimation():
    flow_fw = torch.zeros(1, 2, 2, 2)
    flow_fw[0, 0, 0, 0] = 1
    flow_bw = torch.zeros(1, 2, 2, 2)
    flow_bw[0, 0, 0, 0] = -1

    occ_fw = torch.ones(1, 1, 2, 2)
    occ_fw[0, 0, 0, 0] = 0.
    occ_bw = torch.ones(1, 1, 2, 2)
    occ_bw[0, 0, 0, 0] = 0.

    # test invalid mode
    with pytest.raises(AssertionError):
        occlusion_estimation(flow_fw, flow_bw, mode='a')

    # test forward-backward consistency
    occ = occlusion_estimation(
        flow_fw,
        flow_bw,
        mode='consistency',
        warp_cfg=dict(type='Warp', align_corners=True))

    assert occ['occ_fw']
