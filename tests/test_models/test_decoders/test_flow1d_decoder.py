# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.structures import PixelData
from mmengine.utils import is_list_of

from mmflow.models.decoders.flow1d_decoder import (Flow1DDecoder,
                                                   MotionEncoderFlow1D)
from mmflow.structures import FlowDataSample


@pytest.mark.parametrize('net_type', ['Basic', 'Small'])
def test_motion_encoder_flow1d(net_type):

    # test invalid net_type
    with pytest.raises(AssertionError):
        MotionEncoderFlow1D(net_type='invalid value')

    module = MotionEncoderFlow1D(
        net_type=net_type, conv_cfg=None, norm_cfg=None, act_cfg=None)
    radius = 32

    input_corr = torch.randn((1, 2 * (2 * radius + 1), 56, 56))
    input_flow = torch.randn((1, 2, 56, 56))

    corr_feat = module.corr_net(input_corr)
    flow_feat = module.flow_net(input_flow)
    our_feat = module.out_net(torch.cat([corr_feat, flow_feat], dim=1))

    if net_type == 'Basic':
        assert corr_feat.shape == torch.Size((1, 192, 56, 56))
        assert flow_feat.shape == torch.Size((1, 64, 56, 56))
        assert our_feat.shape == torch.Size((1, 126, 56, 56))
    elif net_type == 'Small':
        assert corr_feat.shape == torch.Size((1, 96, 56, 56))
        assert flow_feat.shape == torch.Size((1, 32, 56, 56))
        assert our_feat.shape == torch.Size((1, 80, 56, 56))


def test_flow1d_decoder():
    model = Flow1DDecoder(
        net_type='Basic',
        radius=32,
        iters=12,
        flow_loss=dict(type='SequenceLoss'))
    mask = torch.ones((1, 64 * 9, 10, 10))
    flow = torch.randn((1, 2, 10, 10))
    assert model._upsample(flow, mask).shape == torch.Size((1, 2, 80, 80))

    feat1 = torch.randn(1, 256, 8, 8)
    feat2 = torch.randn(1, 256, 8, 8)
    h_feat = torch.randn(1, 128, 8, 8)
    cxt_feat = torch.randn(1, 128, 8, 8)
    flow = torch.zeros((1, 2, 8, 8))

    h = 64
    w = 64
    metainfo = dict(img_shape=(h, w, 3), ori_shape=(h, w))
    data_sample = FlowDataSample(metainfo=metainfo)
    data_sample.gt_flow_fw = PixelData(**dict(data=torch.randn(2, h, w)))
    data_samples = [data_sample]

    # test forward function
    out = model(feat1, feat2, flow, h_feat, cxt_feat)
    assert isinstance(out, list)
    assert out[0].shape == torch.Size((1, 2, 64, 64))

    # test loss forward
    loss = model.loss(
        feat1, feat2, flow, h_feat, cxt_feat, data_samples=data_samples)
    assert float(loss['loss_flow']) > 0.

    # test predict forward
    out = model.predict(
        feat1, feat2, flow, h_feat, cxt_feat, data_samples=data_samples)
    assert out[0].pred_flow_fw.shape == (64, 64)
    assert isinstance(out, list) and is_list_of(out, FlowDataSample)
