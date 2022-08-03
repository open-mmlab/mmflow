# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch import Tensor

from mmflow.models.utils.attention1d import Attention1D, AttentionLayer

_feature1 = Tensor([[[[0., 1.], [2., 3.]]]])
_feature2 = Tensor([[[[1., 2.], [3., 4.]]]])

# feature shape (1, 1, 2, 2)
b, c, h, w = _feature1.size()


@pytest.mark.parametrize('y_attention', [True, False])
def test_attentionLayer(y_attention):
    attn_layer = AttentionLayer(in_channels=c, y_attention=y_attention)

    # ground truth
    gt_out_x = Tensor([[[[1.9526, 1.9933], [3.9991, 3.9999]]]])
    gt_scores_x = Tensor([[[[4.7426e-02, 9.5257e-01], [6.6929e-03,
                                                       9.9331e-01]],
                           [[9.1105e-04, 9.9909e-01], [1.2339e-04,
                                                       9.9988e-01]]]])
    gt_out_y = Tensor([[[[2.9951, 3.9999], [3.0000, 4.0000]]]])
    gt_scores_y = Tensor([[[[2.4726e-03, 9.9753e-01], [8.3153e-07,
                                                       1.0000e+00]],
                           [[4.5398e-05, 9.9995e-01], [1.5230e-08,
                                                       1.0000e+00]]]])

    # setting weights
    attn_layer.query_conv.weight.data = Tensor([[[[1.]]]])
    attn_layer.query_conv.bias.data = Tensor([1.5])
    attn_layer.key_conv.weight.data = Tensor([[[[2.]]]])
    attn_layer.key_conv.bias.data = Tensor([-1.])

    out, scores = attn_layer(_feature1, _feature2, None, None)
    assert out.size() == (b, c, h, w)
    if y_attention:
        assert scores.size() == (b, w, h, h)
        assert torch.allclose(out, gt_out_y, atol=1e-04)
        assert torch.allclose(scores, gt_scores_y, atol=1e-4)

    else:
        assert scores.size() == (b, h, w, w)
        assert torch.allclose(out, gt_out_x, atol=1e-04)
        assert torch.allclose(scores, gt_scores_x, atol=1e-4)


@pytest.mark.parametrize('y_attention', [True, False])
@pytest.mark.parametrize('double_cross_attn', [True, False])
def test_attention1d(y_attention, double_cross_attn):
    # ground truth
    gt_out_x = Tensor([[[[1.9991, 1.9999], [3.9991, 3.9999]]]])
    gt_scores_x = Tensor([[[[9.2010e-04, 9.9908e-01], [1.2342e-04,
                                                       9.9988e-01]],
                           [[9.1105e-04, 9.9909e-01], [1.2339e-04,
                                                       9.9988e-01]]]])
    gt_selfattn_y = Tensor([[[[1.9951, 2.9999], [2.0000, 3.0000]]]])
    gt_out_y = Tensor([[[[2.9999, 3.9999], [3.0000, 4.0000]]]])
    gt_scores_y = Tensor([[[[5.4881e-05, 9.9995e-01], [1.5286e-08,
                                                       1.0000e+00]],
                           [[4.6630e-05, 9.9995e-01], [1.5237e-08,
                                                       1.0000e+00]]]])
    gt_selfattn_x = Tensor([[[[0.9526, 0.9933], [2.9991, 2.9999]]]])
    gt_singleattn_out_y = Tensor([[[[2.9951, 3.9999], [3.0000, 4.0000]]]])
    gt_singleattn_scores_y = Tensor([[[[2.4726e-03, 9.9753e-01],
                                       [8.3153e-07, 1.0000e+00]],
                                      [[4.5398e-05, 9.9995e-01],
                                       [1.5230e-08, 1.0000e+00]]]])
    gt_singleattn_out_x = Tensor([[[[1.9526, 1.9933], [3.9991, 3.9999]]]])
    gt_singleattn_scores_x = Tensor([[[[4.7426e-02, 9.5257e-01],
                                       [6.6929e-03, 9.9331e-01]],
                                      [[9.1105e-04, 9.9909e-01],
                                       [1.2339e-04, 9.9988e-01]]]])

    # initialize Attention1D
    attn = Attention1D(
        in_channels=c,
        y_attention=y_attention,
        double_cross_attn=double_cross_attn)

    # setting weights
    attn.cross_attn.query_conv.weight.data = Tensor([[[[1.]]]])
    attn.cross_attn.query_conv.bias.data = Tensor([1.5])
    attn.cross_attn.key_conv.weight.data = Tensor([[[[2.]]]])
    attn.cross_attn.key_conv.bias.data = Tensor([-1.])
    if double_cross_attn:
        attn.self_attn.query_conv.weight.data = Tensor([[[[1.]]]])
        attn.self_attn.query_conv.bias.data = Tensor([1.5])
        attn.self_attn.key_conv.weight.data = Tensor([[[[2.]]]])
        attn.self_attn.key_conv.bias.data = Tensor([-1.])
        selfattn, _ = attn.self_attn(_feature1, _feature1, None, None)

    # test attention
    out, scores = attn.forward(_feature1, _feature2, None, None)
    assert out.size() == (b, c, h, w)
    if double_cross_attn:
        if y_attention:
            assert selfattn.size() == (b, c, h, w)
            assert torch.allclose(selfattn, gt_selfattn_x, atol=1e-04)
            assert scores.size() == (b, w, h, h)
            assert torch.allclose(out, gt_out_y, atol=1e-04)
            assert torch.allclose(scores, gt_scores_y, atol=1e-04)
        else:
            assert selfattn.size() == (b, c, h, w)
            assert torch.allclose(selfattn, gt_selfattn_y, atol=1e-04)
            assert scores.size() == (b, h, w, w)
            assert torch.allclose(out, gt_out_x, atol=1e-04)
            assert torch.allclose(scores, gt_scores_x, atol=1e-04)
    else:
        if y_attention:
            assert scores.size() == (b, w, h, h)
            assert torch.allclose(out, gt_singleattn_out_y, atol=1e-04)
            assert torch.allclose(scores, gt_singleattn_scores_y, atol=1e-04)
        else:
            assert scores.size() == (b, h, w, w)
            assert torch.allclose(out, gt_singleattn_out_x, atol=1e-04)
            assert torch.allclose(scores, gt_singleattn_scores_x, atol=1e-04)
