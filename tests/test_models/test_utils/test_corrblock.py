# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.utils import CorrBlock


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('scaled', [True, False])
def test_corr_block(scaled):
    feat1 = torch.randn(1, 10, 10, 10).cuda()
    feat2 = torch.randn(1, 10, 10, 10).cuda()
    corr_block_cfg = dict(
        corr_cfg=dict(type='Correlation', max_displacement=1, padding=0))

    out = CorrBlock(**corr_block_cfg, scaled=scaled)(feat1, feat2)

    assert out.shape == torch.Size((1, 9, 10, 10))
