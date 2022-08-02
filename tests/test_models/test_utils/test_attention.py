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
    feature_x_gt = Tensor([[[[1.9526, 1.9933], [3.9991, 3.9999]]]])
    attention_x_gt = Tensor([[[[4.7426e-02, 9.5257e-01],
                               [6.6929e-03, 9.9331e-01]],
                              [[9.1105e-04, 9.9909e-01],
                               [1.2339e-04, 9.9988e-01]]]])
    feature_y_gt = Tensor([[[[2.9951, 3.9999], [3.0000, 4.0000]]]])
    attention_y_gt = Tensor([[[[2.4726e-03, 9.9753e-01],
                               [8.3153e-07, 1.0000e+00]],
                              [[4.5398e-05, 9.9995e-01],
                               [1.5230e-08, 1.0000e+00]]]])

    # setting weights
    attn_layer.query_conv.weight.data = Tensor([[[[1.]]]])
    attn_layer.query_conv.bias.data = Tensor([1.5])
    attn_layer.key_conv.weight.data = Tensor([[[[2.]]]])
    attn_layer.key_conv.bias.data = Tensor([-1.])

    out, scores = attn_layer(_feature1, _feature2, None, None)
    assert out.size() == (b, c, h, w)
    if y_attention:
        assert scores.size() == (b, w, h, h)
        assert torch.allclose(out, feature_y_gt, atol=1e-04)
        assert torch.allclose(scores, attention_y_gt, atol=1e-4)

    else:
        assert scores.size() == (b, h, w, w)
        assert torch.allclose(out, feature_x_gt, atol=1e-04)
        assert torch.allclose(scores, attention_x_gt, atol=1e-4)


@pytest.mark.parametrize('y_attention', [True, False])
def test_attention1d(y_attention):
    # ground truth
    feature2_x_gt = Tensor([[[[1.9991, 1.9999], [3.9991, 3.9999]]]])
    attention_x_gt = Tensor([[[[9.2010e-04, 9.9908e-01],
                               [1.2342e-04, 9.9988e-01]],
                              [[9.1105e-04, 9.9909e-01],
                               [1.2339e-04, 9.9988e-01]]]])
    selfattn_y_gt = Tensor([[[[1.9951, 2.9999], [2.0000, 3.0000]]]])
    feature2_y_gt = Tensor([[[[2.9999, 3.9999], [3.0000, 4.0000]]]])
    attention_y_gt = Tensor([[[[5.4881e-05, 9.9995e-01],
                               [1.5286e-08, 1.0000e+00]],
                              [[4.6630e-05, 9.9995e-01],
                               [1.5237e-08, 1.0000e+00]]]])
    selfattn_x_gt = Tensor([[[[0.9526, 0.9933], [2.9991, 2.9999]]]])

    # initialize Attention1D
    attn = Attention1D(
        in_channels=c, y_attention=y_attention, double_cross_attn=True)

    # setting weights
    attn.self_attn.query_conv.weight.data = Tensor([[[[1.]]]])
    attn.cross_attn.query_conv.weight.data = Tensor([[[[1.]]]])
    attn.self_attn.query_conv.bias.data = Tensor([1.5])
    attn.cross_attn.query_conv.bias.data = Tensor([1.5])
    attn.self_attn.key_conv.weight.data = Tensor([[[[2.]]]])
    attn.cross_attn.key_conv.weight.data = Tensor([[[[2.]]]])
    attn.self_attn.key_conv.bias.data = Tensor([-1.])
    attn.cross_attn.key_conv.bias.data = Tensor([-1.])

    # test attention
    selfattn, _ = attn.self_attn(_feature1, _feature1, None, None)
    out, attention = attn.forward(_feature1, _feature2, None, None)
    assert out.size() == (b, c, h, w)
    if y_attention:
        assert selfattn.size() == (b, c, h, w)
        assert torch.allclose(selfattn, selfattn_x_gt, atol=1e-04)
        assert attention.size() == (b, w, h, h)
        assert torch.allclose(out, feature2_y_gt, atol=1e-04)
        assert torch.allclose(attention, attention_y_gt, atol=1e-04)
    else:
        assert selfattn.size() == (b, c, h, w)
        assert torch.allclose(selfattn, selfattn_y_gt, atol=1e-04)
        assert attention.size() == (b, h, w, w)
        assert torch.allclose(out, feature2_x_gt, atol=1e-04)
        assert torch.allclose(attention, attention_x_gt, atol=1e-04)
