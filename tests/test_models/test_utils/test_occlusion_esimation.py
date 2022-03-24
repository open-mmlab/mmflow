# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.utils import occlusion_estimation


def test_occlusion_estimation():
    """Test occ estimation."""
    """
    img1           img2

    | A | B | E |  | G | A | B |
    -------------  -------------
    | C | D | F |  | H | C | D |

    flow_fw                  flow_bw
    |(1, 0)|(1, 0)|(1, 0)|   |(-1, 0)|(-1, 0)|(-1, 0)|
    ----------------------   -------------------------
    |(1, 0)|(1, 0)|(1, 0)|   |(-1, 0)|(-1, 0)|(-1, 0)|

    occ_fw         occ_bw
    | 1 | 1 | 0 |  | 0 | 1 | 1 |
    -------------  -------------
    | 1 | 1 | 0 |  | 0 | 1 | 1 |
    """
    H = 2
    W = 3
    flow_fw = torch.zeros(4, 2, H, W)
    flow_fw[:, 0, ...] = 1
    flow_bw = -flow_fw.clone()

    occ_fw = torch.ones(4, 1, H, W)
    occ_fw[..., -1] = 0.
    occ_bw = torch.ones(4, 1, H, W)
    occ_bw[..., 0] = 0.

    # test invalid mode
    with pytest.raises(AssertionError):
        occlusion_estimation(flow_fw, flow_bw, mode='a')

    # test forward-backward consistency
    occ = occlusion_estimation(
        flow_fw,
        flow_bw,
        mode='consistency',
        warp_cfg=dict(type='Warp', align_corners=True))
    assert torch.all(occ['occ_fw'] == occ_fw)
    assert torch.all(occ['occ_bw'] == occ_bw)

    # test fb_abs
    occ = occlusion_estimation(
        flow_fw,
        flow_bw,
        mode='fb_abs',
        warp_cfg=dict(type='Warp', align_corners=True),
        diff=1.)
    assert torch.all(occ['occ_fw'] == occ_fw)
    assert torch.all(occ['occ_bw'] == occ_bw)

    # test range map
    occ = occlusion_estimation(flow_fw, flow_bw, mode='range_map')
    assert torch.all(occ['occ_fw'] == occ_fw)
    assert torch.all(occ['occ_bw'] == occ_bw)
