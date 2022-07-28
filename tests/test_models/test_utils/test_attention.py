# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmflow.models.utils.attention import Attention1D


def test_attention():
    feature1 = torch.randn(1, 3, 2, 4)
    feature2 = torch.randn(1, 3, 2, 4)
    b, c, h, w = feature1.size()

    #  test cross attention y
    attn_y = Attention1D(in_channels=3, y_attention=True)
    # test x's self attention first
    selfattn_x, _ = attn_y.self_attn(feature1, None, None)
    assert selfattn_x.size() == (b, c, h, w)
    feature2_y, attention_y = attn_y.forward(feature1, feature2, None, None)
    assert feature2_y.size() == (b, c, h, w)
    assert attention_y.size() == (b, w, h, h)

    #  test cross attention x
    attn_x = Attention1D(in_channels=3, y_attention=False)
    # test y's self attention first
    selfattn_y, _ = attn_x.self_attn(feature1, None, None)
    assert selfattn_y.size() == (b, c, h, w)
    feature2_x, attention_x = attn_x.forward(feature1, feature2, None, None)
    assert feature2_x.size() == (b, c, h, w)
    assert attention_x.size() == (b, h, w, w)
