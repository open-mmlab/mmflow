# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt

import pytest
import torch

from mmflow.models.utils import CorrBlock


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('scaled', [True, False])
def test_corr_block(scaled):
    feat1 = torch.randn(1, 3, 10, 10).cuda()
    feat2 = torch.randn(1, 3, 10, 10).cuda()
    corr_block_cfg = dict(
        corr_cfg=dict(type='Correlation', max_displacement=1, padding=0))

    out = CorrBlock(**corr_block_cfg, scaled=scaled)(feat1, feat2)

    assert out.shape == torch.Size((1, 9, 10, 10))

    with pytest.raises(AssertionError):
        CorrBlock(**corr_block_cfg, scaled=scaled, scale_mode='test')

    if scaled:

        out = CorrBlock(**corr_block_cfg, scaled=False)(feat1, feat2)

        out_scaled_dimension = CorrBlock(
            **corr_block_cfg, scaled=scaled, scale_mode='dimension')(feat1,
                                                                     feat2)
        out_scaled_sqrtdimension = CorrBlock(
            **corr_block_cfg, scaled=scaled,
            scale_mode='sqrt dimension')(feat1, feat2)

        # test scaled by dimension and sqrt dimension
        assert torch.allclose(out, out_scaled_dimension * 3)
        assert torch.allclose(out, out_scaled_sqrtdimension * sqrt(3))
