# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.encoders import RAFTEncoder


def test_raftbackbone():

    # test net_type in RAFT backbone
    with pytest.raises(KeyError):
        # test invalid net_type
        RAFTEncoder(
            in_channels=3,
            out_channels=256,
            net_type=1,
            norm_cfg=dict(type='IN'),
        )
    model = RAFTEncoder(
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='IN'))

    model.train()
    img = torch.randn(1, 3, 224, 224)
    feat = model(img)
    assert feat.shape == torch.Size([1, 256, 28, 28])
