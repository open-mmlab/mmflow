# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmflow.models.utils.attention1d import Attention1D, AttentionLayer


def test_attentionLayer():
    feature1 = torch.tensor([[[[0., 1.], [2., 3.]]]])
    feature2 = torch.tensor([[[[1., 2.], [3., 4.]]]])

    # feature shape (1, 1, 2, 2)
    b, c, h, w = feature1.size()
    attn_layer_y = AttentionLayer(in_channels=c, y_attention=True)
    attn_layer_x = AttentionLayer(in_channels=c, y_attention=False)

    # setting weights
    attn_layer_x.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_layer_y.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_layer_x.query_conv.bias.data = torch.Tensor([1.])
    attn_layer_y.query_conv.bias.data = torch.Tensor([1.])
    attn_layer_x.key_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_layer_y.key_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_layer_x.key_conv.bias.data = torch.Tensor([1.])
    attn_layer_y.key_conv.bias.data = torch.Tensor([1.])

    feature_y, attention_y = attn_layer_y(feature1, feature2, None, None)
    feature_x, attention_x = attn_layer_x(feature1, feature2, None, None)
    assert (b, c, h, w) == feature_x.size()
    assert (b, c, h, w) == feature_y.size()


def test_attention1d():
    feature1 = torch.tensor([[[[0., 1.], [2., 3.]]]])
    feature2 = torch.tensor([[[[1., 2.], [3., 4.]]]])

    # feature shape (1, 1, 2, 2)
    b, c, h, w = feature1.size()

    # test cross attention y
    attn_y = Attention1D(
        in_channels=c, y_attention=True, double_cross_attn=True)

    # setting weights
    attn_y.self_attn.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_y.cross_attn.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_y.self_attn.query_conv.bias.data = torch.Tensor([1.])
    attn_y.cross_attn.query_conv.bias.data = torch.Tensor([1.])
    attn_y.self_attn.key_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_y.cross_attn.key_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_y.self_attn.key_conv.bias.data = torch.Tensor([1.])
    attn_y.cross_attn.key_conv.bias.data = torch.Tensor([1.])

    # test x's self attention first
    selfattn_x, _ = attn_y.self_attn(feature1, feature1, None, None)
    assert selfattn_x.size() == (b, c, h, w)

    # cross attention on y direction
    feature2_y, attention_y = attn_y.forward(feature1, feature2, None, None)
    assert feature2_y.size() == (b, c, h, w)
    assert attention_y.size() == (b, w, h, h)

    #  test cross attention x
    attn_x = Attention1D(in_channels=c, y_attention=False)

    # setting weights
    attn_x.self_attn.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_x.cross_attn.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_x.self_attn.query_conv.bias.data = torch.Tensor([1.])
    attn_x.cross_attn.query_conv.bias.data = torch.Tensor([1.])
    attn_x.self_attn.key_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_x.cross_attn.key_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_x.self_attn.key_conv.bias.data = torch.Tensor([1.])
    attn_x.cross_attn.key_conv.bias.data = torch.Tensor([1.])

    # test y's self attention first
    selfattn_y, _ = attn_x.self_attn(feature1, feature1, None, None)
    assert selfattn_y.size() == (b, c, h, w)

    # cross attention on x direction
    feature2_x, attention_x = attn_x.forward(feature1, feature2, None, None)
    assert feature2_x.size() == (b, c, h, w)
    assert attention_x.size() == (b, h, w, w)
