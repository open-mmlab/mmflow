# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import pytest
import torch
from mmengine.data import PixelData

from mmflow.models.decoders.flownet_decoder import (BasicBlock, DeconvModule,
                                                    FlowNetCDecoder,
                                                    FlowNetSDecoder)
from mmflow.structures import FlowDataSample


def _get_test_data_cuda(
        _channels=dict(
            level1=64,
            level2=128,
            level3=256,
            level4=512,
            level5=512,
            level6=1024),
        w=64,
        h=64):

    metainfo = dict(img_shape=(h, w, 3), ori_shape=(h, w))
    data_sample = FlowDataSample(metainfo=metainfo)
    data_sample.gt_flow_fw = PixelData(**dict(data=torch.randn(2, h, w)))
    batch_data_samples = [data_sample.cuda()]

    feat = dict()
    for level, ch in _channels.items():
        feat[level] = torch.randn(1, ch, h, w).cuda()
        w = w // 2
        h = h // 2
    return feat, batch_data_samples, metainfo


def _get_test_data_cpu(
        _channels=dict(
            level1=64,
            level2=128,
            level3=256,
            level4=512,
            level5=512,
            level6=1024),
        w=64,
        h=64):
    metainfo = dict(img_shape=(h, w, 3), ori_shape=(h, w))
    data_sample = FlowDataSample(metainfo=metainfo)
    data_sample.gt_flow_fw = PixelData(**dict(data=torch.randn(2, h, w)))
    batch_data_samples = [data_sample]

    feat = dict()
    for level, ch in _channels.items():
        feat[level] = torch.randn(1, ch, h, w)
        w = w // 2
        h = h // 2
    return feat, batch_data_samples, metainfo


def test_deconv_module():
    layer = DeconvModule(in_channels=3, out_channels=4)

    input = torch.randn((1, 3, 10, 10))

    output = layer(input)

    assert output.shape == torch.Size((1, 4, 20, 20))


@pytest.mark.parametrize(('out_channels', 'inter_channels'), [(None, None),
                                                              (20, None),
                                                              (None, 20),
                                                              (20, 30)])
def test_basic_block(out_channels, inter_channels):
    in_ch = 10
    input = torch.randn(1, in_ch, 56, 56)

    submodule = BasicBlock(
        in_channels=in_ch,
        pred_channels=2,
        out_channels=out_channels,
        inter_channels=inter_channels)
    output = submodule(input)
    if out_channels is None:
        assert not submodule.up_sample
        assert output[1] is None and output[2] is None  # no upflow and upfeat
    else:
        assert output[1].shape == torch.Size((1, 2, 56 * 2, 56 * 2))
        assert output[2].shape == torch.Size((1, out_channels, 56 * 2, 56 * 2))

    if inter_channels is None:
        # test without inter_conv
        assert isinstance(submodule.pred_out, torch.nn.Conv2d)
    else:
        # test with inter_conv
        assert len(submodule.pred_out) == 2
        assert submodule.pred_out[0].out_channels == inter_channels


@pytest.mark.parametrize(
    ('in_channels', 'out_channels', 'inter_channels'),
    [  # flownets decoder setting
        (
            dict(level6=1024, level5=1026, level4=770, level3=386, level2=194),
            dict(level6=512, level5=256, level4=128, level3=64),
            None,
        ),
        # flownetSD decoder setting
        (
            dict(level6=1024, level5=1026, level4=770, level3=386, level2=194),
            dict(level6=512, level5=256, level4=128, level3=64),
            dict(level5=512, level4=256, level3=128, level2=64),
        )
    ])
def test_flownets_decoder(in_channels, out_channels, inter_channels):

    model = FlowNetSDecoder(
        in_channels=in_channels,
        out_channels=out_channels,
        inter_channels=inter_channels,
        deconv_bias=True,
        pred_bias=True,
        upsample_bias=True,
        flow_loss=dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights={
                'level2': 0.005,
                'level3': 0.01,
                'level4': 0.02,
                'level5': 0.08,
                'level6': 0.32
            }))

    feat, batch_data_samples, metainfo = _get_test_data_cpu()

    # test multi-levels flow forward
    out = model(feat)

    assert isinstance(out, dict)
    assert out['level6'].shape == torch.Size((1, 2, 2, 2))
    assert out['level5'].shape == torch.Size((1, 2, 4, 4))
    assert out['level4'].shape == torch.Size((1, 2, 8, 8))
    assert out['level3'].shape == torch.Size((1, 2, 16, 16))
    assert out['level2'].shape == torch.Size((1, 2, 32, 32))

    # test loss forward
    loss = model.loss(feat, batch_data_samples=batch_data_samples)
    assert float(loss['loss_flow']) > 0

    # test predict forward
    out = model.predict(feat, batch_img_metas=[metainfo])
    assert isinstance(out, list) and mmcv.is_list_of(out, FlowDataSample)
    assert out[0].pred_flow_fw.shape == (64, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_flownetc_decoder():
    model = FlowNetCDecoder(
        in_channels=dict(
            level6=1024, level5=1026, level4=770, level3=386, level2=194),
        out_channels=dict(level6=512, level5=256, level4=128, level3=64),
        flow_loss=dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights={
                'level2': 0.005,
                'level3': 0.01,
                'level4': 0.02,
                'level5': 0.08,
                'level6': 0.32
            })).cuda()

    feat1, batch_data_samples, metainfo = _get_test_data_cuda(
        dict(level1=64, level2=128, level3=256))
    corr_feat, _, _ = _get_test_data_cuda(
        dict(level3=256, level4=512, level5=512, level6=1024), w=16, h=16)

    # test multi-levels flow
    out = model(feat1, corr_feat)
    assert isinstance(out, dict)
    assert out['level6'].shape == torch.Size((1, 2, 2, 2))
    assert out['level5'].shape == torch.Size((1, 2, 4, 4))
    assert out['level4'].shape == torch.Size((1, 2, 8, 8))
    assert out['level3'].shape == torch.Size((1, 2, 16, 16))
    assert out['level2'].shape == torch.Size((1, 2, 32, 32))

    # test loss forward
    loss = model.loss(feat1, corr_feat, batch_data_samples=batch_data_samples)
    assert float(loss['loss_flow']) > 0

    # test predict forward
    out = model.predict(feat1, corr_feat, batch_img_metas=[metainfo])
    assert isinstance(out, list) and mmcv.is_list_of(out, FlowDataSample)
    assert out[0].pred_flow_fw.shape == (64, 64)
