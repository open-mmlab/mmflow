# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import pytest
import torch
from mmengine.data import PixelData

from mmflow.models.decoders.gma_decoder import (Aggregate, Attention,
                                                GMADecoder, RelPosEmb)
from mmflow.structures import FlowDataSample


def test_relposemb():
    H = 8
    W = 6
    heads = 2
    head_channels = 4
    max_pos_size = 50

    rel_pos_emb = RelPosEmb(
        max_pos_size=max_pos_size, head_channels=head_channels)

    q = torch.randn((1, heads, H, W, head_channels))

    assert rel_pos_emb(q).shape == (1, heads, H * W, H * W)


@pytest.mark.parametrize(
    ('max_pos_size', 'position_only'),
    [[None, True], [100, True], [None, False], [100, False]])
def test_attention(max_pos_size, position_only):

    H = 8
    W = 6
    channels = 3
    heads = 2
    head_channels = 4

    if max_pos_size is None and position_only:
        # test Runtime when position_only = True and max_pos_size = None
        with pytest.raises(RuntimeError):
            attn = Attention(
                in_channels=channels,
                heads=heads,
                head_channels=head_channels,
                position_only=position_only,
                max_pos_size=max_pos_size)
    else:
        attn = Attention(
            in_channels=channels,
            heads=heads,
            head_channels=head_channels,
            max_pos_size=max_pos_size,
            position_only=position_only)

        feature = torch.randn((1, channels, H, W))

        out = attn(feature)

        assert out.shape == (1, heads, H * W, H * W)


def test_aggregate():
    H = 8
    W = 6
    channels = 3
    heads = 2
    head_channels = 4

    aggr = Aggregate(
        in_channels=channels, heads=heads, head_channels=head_channels)

    attn = torch.randn((1, heads, H * W, H * W))
    feature = torch.randn(1, channels, H, W)

    out = aggr(attn, feature)

    assert out.shape == (1, channels, H, W)


@pytest.mark.parametrize(
    ('max_pos_size', 'position_only'),
    [[None, True], [100, True], [None, False], [100, False]])
def test_gmadecoder(max_pos_size, position_only):

    heads = 2
    motion_channels = 128
    if max_pos_size is None and position_only:
        # test Runtime when position_only = True and max_pos_size = None
        with pytest.raises(RuntimeError):
            GMADecoder(
                heads=heads,
                motion_channels=motion_channels,
                position_only=position_only,
                max_pos_size=max_pos_size,
                net_type='Basic',
                num_levels=4,
                radius=4,
                iters=12,
                flow_loss=dict(type='SequenceLoss'))
    else:
        model = GMADecoder(
            heads=heads,
            motion_channels=motion_channels,
            position_only=position_only,
            max_pos_size=max_pos_size,
            net_type='Basic',
            num_levels=4,
            radius=4,
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
        assert isinstance(out, list) and mmcv.is_list_of(out, FlowDataSample)
