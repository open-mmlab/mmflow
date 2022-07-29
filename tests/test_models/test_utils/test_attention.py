# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
from torch import tensor

from mmflow.models.utils.attention import Attention1D


def test_attention():
    feature1 = torch.tensor([[[[1.4851, -0.0514, -0.9695, -0.1100],
                               [-0.3388, -1.3190, 0.1436, -0.6746]],
                              [[-0.4149, 0.5351, -0.5995, -0.8030],
                               [2.2214, -1.0711, -0.9920, -1.4901]],
                              [[-0.0287, 0.2969, -0.4293, -0.6187],
                               [-0.1972, 0.0687, -0.5300, 0.1617]]]])
    feature2 = torch.tensor([[[[-0.4095, 1.8635, -0.7602, 0.4474],
                               [1.1612, -0.0092, -0.7034, -0.2123]],
                              [[1.8637, 0.8363, 1.5985, 0.5270],
                               [1.3758, 0.2204, -0.4623, 1.6197]],
                              [[-1.6217, 1.8201, 0.9509, -2.5758],
                               [-1.0160, -0.2388, -0.3180, 0.1021]]]])

    # feature shape (1, 3, 2, 4)
    b, c, h, w = feature1.size()

    #  test cross attention y
    attn_y = Attention1D(in_channels=3, y_attention=True)
    state_dict = OrderedDict([
        ('self_attn.query_conv.weight',
         tensor([[[[-0.0962]], [[-0.3709]], [[-0.1936]]],
                 [[[0.4256]], [[-0.2761]], [[0.5265]]],
                 [[[-0.5304]], [[-0.0033]], [[-0.7552]]]])),
        ('self_attn.query_conv.bias', tensor([0.5679, -0.5169, -0.5476])),
        ('self_attn.key_conv.weight',
         tensor([[[[0.9883]], [[-0.7814]], [[0.8251]]],
                 [[[-0.3207]], [[-0.7325]], [[-0.5771]]],
                 [[[-0.7312]], [[-0.1323]], [[-0.2281]]]])),
        ('self_attn.key_conv.bias', tensor([-0.1655, -0.4254, -0.3746])),
        ('cross_attn.query_conv.weight',
         tensor([[[[0.5491]], [[-0.7741]], [[-0.2849]]],
                 [[[-0.6808]], [[-0.0640]], [[0.3206]]],
                 [[[0.8917]], [[0.5406]], [[-0.3645]]]])),
        ('cross_attn.query_conv.bias', tensor([0.1521, -0.3843, -0.5711])),
        ('cross_attn.key_conv.weight',
         tensor([[[[0.7493]], [[-0.5769]], [[0.8654]]],
                 [[[0.7737]], [[0.4463]], [[-0.7268]]],
                 [[[-0.2483]], [[0.3091]], [[0.3188]]]])),
        ('cross_attn.key_conv.bias', tensor([0.2244, 0.3538, -0.0416]))
    ])
    attn_y.load_state_dict(state_dict)

    # test x's self attention first
    selfattn_x, _ = attn_y.self_attn(feature1, feature1, None, None)
    assert selfattn_x.size() == (b, c, h, w)

    # cross attention on y direction
    feature2_y, attention_y = attn_y.forward(feature1, feature2, None, None)
    assert feature2_y.size() == (b, c, h, w)
    assert attention_y.size() == (b, w, h, h)

    #  test cross attention x
    attn_x = Attention1D(in_channels=3, y_attention=False)
    state_dict = OrderedDict([
        ('self_attn.query_conv.weight',
         tensor([[[[-0.2707]], [[0.8776]], [[0.2155]]],
                 [[[0.4508]], [[-0.3846]], [[0.4017]]],
                 [[[0.4647]], [[-0.5446]], [[0.8714]]]])),
        ('self_attn.query_conv.bias', tensor([-0.3693, -0.0481, -0.4405])),
        ('self_attn.key_conv.weight',
         tensor([[[[0.2769]], [[-0.6740]], [[0.0146]]],
                 [[[0.2338]], [[-0.7720]], [[-0.9872]]],
                 [[[0.0912]], [[-0.8806]], [[-0.9097]]]])),
        ('self_attn.key_conv.bias', tensor([0.4059, -0.5126, -0.3617])),
        ('cross_attn.query_conv.weight',
         tensor([[[[-0.9078]], [[0.1079]], [[0.5150]]],
                 [[[0.4239]], [[-0.8712]], [[-0.9505]]],
                 [[[-0.0483]], [[-0.9157]], [[-0.0928]]]])),
        ('cross_attn.query_conv.bias', tensor([0.1816, 0.3177, -0.3526])),
        ('cross_attn.key_conv.weight',
         tensor([[[[-0.6112]], [[-0.1017]], [[-0.9844]]],
                 [[[0.7257]], [[-0.8277]], [[0.3744]]],
                 [[[-0.9275]], [[-0.9794]], [[-0.8065]]]])),
        ('cross_attn.key_conv.bias', tensor([-0.0744, -0.3446, 0.2543]))
    ])
    attn_x.load_state_dict(state_dict)

    # test y's self attention first
    selfattn_y, _ = attn_x.self_attn(feature1, feature1, None, None)
    assert selfattn_y.size() == (b, c, h, w)

    # cross attention on x direction
    feature2_x, attention_x = attn_x.forward(feature1, feature2, None, None)
    assert feature2_x.size() == (b, c, h, w)
    assert attention_x.size() == (b, h, w, w)
