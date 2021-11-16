# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.models.decoders.irrpwc_decoder import (IRRCorrBlock,
                                                   IRRFlowDecoder,
                                                   IRROccDecoder,
                                                   IRRPWCDecoder)


def _get_test_data(
        _channels=dict(
            level1=64,
            level2=128,
            level3=256,
            level4=512,
            level5=512,
            level6=1024),
        w=32,
        h=32):
    feat = dict()

    for level, ch in _channels.items():
        feat[level] = torch.randn(1, ch, h, w).cuda()
        w = w // 2
        h = h // 2
    return feat


def test_irr_flow_decoder():
    in_channels = 115
    feat_channels = (128, 128, 96, 64, 32)

    module = IRRFlowDecoder(
        in_channels=in_channels,
        feat_channels=feat_channels,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None)

    input = torch.randn(1, in_channels, 56, 56)

    output = module(input)

    assert output[1].shape == torch.Size((1, 2, 56, 56))
    assert output[0].shape == torch.Size(
        (1, in_channels + sum(feat_channels), 56, 56))


def test_irr_occ_head():
    in_channels = 114
    feat_channels = (128, 128, 96, 64, 32)

    module = IRROccDecoder(
        in_channels=in_channels,
        feat_channels=feat_channels,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None)
    input = torch.randn(1, in_channels, 56, 56)

    output = module(input)

    assert output[1].shape == torch.Size((1, 1, 56, 56))
    assert output[0].shape == torch.Size(
        (1, in_channels + sum(feat_channels), 56, 56))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('in_channels', (64, 32))
def test_irr_corr_block(in_channels):
    out_channels = 32
    model = IRRCorrBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        corr_cfg=dict(type='Correlation', max_displacement=4),
        scaled=True,
        warp_cfg=dict(type='Warp', align_corners=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)).cuda()

    if in_channels == 32:
        assert isinstance(model.conv_1x1, torch.nn.Sequential)

    feat1 = torch.randn(1, in_channels, 12, 12).cuda()
    feat2 = torch.randn(1, in_channels, 12, 12).cuda()

    corr_f, feat1_, corr_b, feat2_ = model(feat1, feat2)

    assert corr_f.shape == torch.Size((1, 81, 12, 12))
    assert corr_b.shape == torch.Size((1, 81, 12, 12))
    assert feat1_.shape == torch.Size((1, out_channels, 12, 12))
    assert feat2_.shape == torch.Size((1, out_channels, 12, 12))

    flow_f = torch.randn(1, 2, 12, 12).cuda()
    flow_b = torch.randn(1, 2, 12, 12).cuda()

    corr_f, feat1_, corr_b, feat2_ = model(feat1, feat2, flow_f, flow_b)

    assert corr_f.shape == torch.Size((1, 81, 12, 12))
    assert corr_b.shape == torch.Size((1, 81, 12, 12))
    assert feat1_.shape == torch.Size((1, out_channels, 12, 12))
    assert feat2_.shape == torch.Size((1, out_channels, 12, 12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_irr_pwc_decoder():

    model = IRRPWCDecoder(
        flow_levels=[
            'level0', 'level1', 'level2', 'level3', 'level4', 'level5',
            'level6'
        ],
        corr_in_channels=dict(
            level2=32, level3=64, level4=96, level5=128, level6=196),
        corr_feat_channels=32,
        flow_decoder_in_channels=115,
        occ_decoder_in_channels=114,
        corr_cfg=dict(type='Correlation', max_displacement=4),
        scaled=True,
        warp_cfg=dict(type='Warp', align_corners=True),
        densefeat_channels=(128, 128, 96, 64, 32),
        flow_post_processor=dict(
            type='ContextNet',
            in_channels=565,
            out_channels=2,
            feat_channels=(128, 128, 128, 96, 64, 32),
            dilations=(1, 2, 4, 8, 16, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        flow_refine=dict(
            type='FlowRefine',
            in_channels=35,
            feat_channels=(128, 128, 64, 64, 32, 32),
            patch_size=3,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        ),
        occ_post_processor=dict(
            type='ContextNet',
            in_channels=563,
            out_channels=1,
            feat_channels=(128, 128, 128, 96, 64, 32),
            dilations=(1, 2, 4, 8, 16, 1),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
        occ_refine=dict(
            type='OccRefine',
            in_channels=65,
            feat_channels=(128, 128, 64, 64, 32, 32),
            patch_size=3,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            warp_cfg=dict(type='Warp', align_corners=True),
        ),
        occ_upsample=dict(
            type='OccShuffleUpsample',
            in_channels=11,
            feat_channels=32,
            infeat_channels=16,
            out_channels=1,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            warp_cfg=dict(type='Warp', align_corners=True, use_mask=True)),
        occ_refined_levels=['level0', 'level1'],
        flow_div=20.,
        occ_loss=dict(
            type='MultiLevelBCE',
            weights=dict(
                level6=0.32,
                level5=0.08,
                level4=0.02,
                level3=0.01,
                level2=0.005,
                level1=0.00125,
                level0=0.0003125),
            reduction='sum'),
        flow_loss=dict(
            type='MultiLevelEPE',
            p=2,
            reduction='sum',
            weights=dict(
                level6=0.32,
                level5=0.08,
                level4=0.02,
                level3=0.01,
                level2=0.005,
                level1=0.00125,
                level0=0.0003125)),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)).cuda()

    input_channels = dict(
        level1=16, level2=32, level3=64, level4=96, level5=128, level6=196)
    feat1 = _get_test_data(input_channels)
    feat2 = _get_test_data(input_channels)

    feat1['level0'] = torch.randn(1, 3, 16 * 4, 16 * 4).cuda()
    feat2['level0'] = torch.randn(1, 3, 16 * 4, 16 * 4).cuda()

    flow_fw_gt = torch.randn(1, 2, 64, 64).cuda()
    flow_bw_gt = torch.randn(1, 2, 64, 64).cuda()
    occ_fw_gt = torch.randn(1, 1, 64, 64).cuda()
    occ_bw_gt = torch.randn(1, 1, 64, 64).cuda()

    # test forward_train out with flow_fw_gt, flow_bw_gt, occ_fw_gt, occ_bw_gt
    loss = model.forward_train(
        feat1,
        feat2,
        flow_fw_gt=flow_fw_gt,
        flow_bw_gt=flow_bw_gt,
        occ_fw_gt=occ_fw_gt,
        occ_bw_gt=occ_bw_gt,
    )
    assert float(loss['loss_flow']) > 0
    assert float(loss['loss_occ']) > 0

    # test forward_train out with flow_gt
    loss = model.forward_train(
        feat1,
        feat2,
        flow_fw_gt=None,
        flow_bw_gt=None,
        occ_fw_gt=None,
        occ_bw_gt=None,
        flow_gt=flow_fw_gt,
    )
    assert float(loss['loss_flow']) > 0
    assert loss.get('loss_occ', None) is None

    # test forward_train out with flow_fw_gt, flow_bw_gt
    loss = model.forward_train(
        feat1, feat2, flow_fw_gt=flow_fw_gt, flow_bw_gt=flow_bw_gt)
    assert float(loss['loss_flow']) > 0
    assert loss.get('loss_occ', None) is None

    # test forward_train out with flow_gt, occ_gt
    loss = model.forward_train(
        feat1, feat2, flow_gt=flow_fw_gt, occ_gt=occ_fw_gt)
    assert float(loss['loss_flow']) > 0
    assert float(loss['loss_occ']) > 0

    # test forward_test
    flow_result = model.forward_test(feat1, feat2, 64, 64)
    assert flow_result[0]['flow_fw'].shape == (64, 64, 2)
    assert flow_result[0]['flow_bw'].shape == (64, 64, 2)
