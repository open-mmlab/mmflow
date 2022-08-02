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

    # ground truth
    feature_x_gt = torch.Tensor([[[[1.9526, 1.9933], [3.9991, 3.9999]]]])
    attention_x_gt = torch.Tensor([[[[4.7426e-02, 9.5257e-01],
                                     [6.6929e-03, 9.9331e-01]],
                                    [[9.1105e-04, 9.9909e-01],
                                     [1.2339e-04, 9.9988e-01]]]])
    feature_y_gt = torch.Tensor([[[[2.9951, 3.9999], [3.0000, 4.0000]]]])
    attention_y_gt = torch.Tensor([[[[2.4726e-03, 9.9753e-01],
                                     [8.3153e-07, 1.0000e+00]],
                                    [[4.5398e-05, 9.9995e-01],
                                     [1.5230e-08, 1.0000e+00]]]])

    # setting weights
    attn_layer_x.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_layer_y.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_layer_x.query_conv.bias.data = torch.Tensor([1.5])
    attn_layer_y.query_conv.bias.data = torch.Tensor([1.5])
    attn_layer_x.key_conv.weight.data = torch.Tensor([[[[2.]]]])
    attn_layer_y.key_conv.weight.data = torch.Tensor([[[[2.]]]])
    attn_layer_x.key_conv.bias.data = torch.Tensor([-1.])
    attn_layer_y.key_conv.bias.data = torch.Tensor([-1.])

    feature_y, attention_y = attn_layer_y(feature1, feature2, None, None)
    feature_x, attention_x = attn_layer_x(feature1, feature2, None, None)
    assert (b, c, h, w) == feature_x.size()
    assert torch.allclose(feature_x, feature_x_gt, atol=1e-04)
    assert torch.allclose(attention_x, attention_x_gt, atol=1e-04)
    assert (b, c, h, w) == feature_y.size()
    assert torch.allclose(feature_y, feature_y_gt, atol=1e-04)
    assert torch.allclose(attention_y, attention_y_gt, atol=1e-04)


def test_attention1d():
    feature1 = torch.tensor([[[[0., 1.], [2., 3.]]]])
    feature2 = torch.tensor([[[[1., 2.], [3., 4.]]]])

    # feature shape (1, 1, 2, 2)
    b, c, h, w = feature1.size()

    # ground truth
    feature2_x_gt = torch.Tensor([[[[1.9991, 1.9999], [3.9991, 3.9999]]]])
    attention_x_gt = torch.Tensor([[[[9.2010e-04, 9.9908e-01],
                                     [1.2342e-04, 9.9988e-01]],
                                    [[9.1105e-04, 9.9909e-01],
                                     [1.2339e-04, 9.9988e-01]]]])
    selfattn_y_gt = torch.Tensor([[[[1.9951, 2.9999], [2.0000, 3.0000]]]])
    feature2_y_gt = torch.Tensor([[[[2.9999, 3.9999], [3.0000, 4.0000]]]])
    attention_y_gt = torch.Tensor([[[[5.4881e-05, 9.9995e-01],
                                     [1.5286e-08, 1.0000e+00]],
                                    [[4.6630e-05, 9.9995e-01],
                                     [1.5237e-08, 1.0000e+00]]]])
    selfattn_x_gt = torch.Tensor([[[[0.9526, 0.9933], [2.9991, 2.9999]]]])

    # test cross attention y
    attn_y = Attention1D(
        in_channels=c, y_attention=True, double_cross_attn=True)

    # setting weights
    attn_y.self_attn.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_y.cross_attn.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_y.self_attn.query_conv.bias.data = torch.Tensor([1.5])
    attn_y.cross_attn.query_conv.bias.data = torch.Tensor([1.5])
    attn_y.self_attn.key_conv.weight.data = torch.Tensor([[[[2.]]]])
    attn_y.cross_attn.key_conv.weight.data = torch.Tensor([[[[2.]]]])
    attn_y.self_attn.key_conv.bias.data = torch.Tensor([-1.])
    attn_y.cross_attn.key_conv.bias.data = torch.Tensor([-1.])

    # test x's self attention first
    selfattn_x, _ = attn_y.self_attn(feature1, feature1, None, None)
    assert selfattn_x.size() == (b, c, h, w)
    assert torch.allclose(selfattn_x, selfattn_x_gt, atol=1e-04)
    # cross attention on y direction
    feature2_y, attention_y = attn_y.forward(feature1, feature2, None, None)
    assert feature2_y.size() == (b, c, h, w)
    assert attention_y.size() == (b, w, h, h)
    assert torch.allclose(feature2_y, feature2_y_gt, atol=1e-04)
    assert torch.allclose(attention_y, attention_y_gt, atol=1e-04)

    #  test cross attention x
    attn_x = Attention1D(in_channels=c, y_attention=False)

    # setting weights
    attn_x.self_attn.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_x.cross_attn.query_conv.weight.data = torch.Tensor([[[[1.]]]])
    attn_x.self_attn.query_conv.bias.data = torch.Tensor([1.5])
    attn_x.cross_attn.query_conv.bias.data = torch.Tensor([1.5])
    attn_x.self_attn.key_conv.weight.data = torch.Tensor([[[[2.]]]])
    attn_x.cross_attn.key_conv.weight.data = torch.Tensor([[[[2.]]]])
    attn_x.self_attn.key_conv.bias.data = torch.Tensor([-1.])
    attn_x.cross_attn.key_conv.bias.data = torch.Tensor([-1.])

    # test y's self attention first
    selfattn_y, _ = attn_x.self_attn(feature1, feature1, None, None)
    assert selfattn_y.size() == (b, c, h, w)
    assert torch.allclose(selfattn_y, selfattn_y_gt, atol=1e-04)
    # cross attention on x direction
    feature2_x, attention_x = attn_x.forward(feature1, feature2, None, None)
    assert feature2_x.size() == (b, c, h, w)
    assert attention_x.size() == (b, h, w, w)
    assert torch.allclose(feature2_x, feature2_x_gt, atol=1e-04)
    assert torch.allclose(attention_x, attention_x_gt, atol=1e-04)
