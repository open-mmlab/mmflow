# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest
import torch
from mmcv.utils import Config

from mmflow.models import build_flow_estimator
from mmflow.models.flow_estimators.base import FlowEstimator


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.parametrize('cfg_file', [
    '../../configs/_base_/models/pwcnet.py',
    '../../configs/_base_/models/maskflownets.py',
    '../../configs/_base_/models/maskflownet.py',
    '../../configs/_base_/models/flownetc.py',
    '../../configs/_base_/models/liteflownet/liteflownet.py',
    '../../configs/_base_/models/flownet2/flownet2cs.py',
    '../../configs/_base_/models/flownet2/flownet2css.py',
    '../../configs/_base_/models/flownet2/flownet2.py',
])
def test_flow_estimator(cfg_file):
    # BaseFlowEstimator has abstract method
    with pytest.raises(TypeError):
        FlowEstimator(init_cfg=None)

    cfg_file = osp.join(osp.dirname(__file__), cfg_file)
    cfg = Config.fromfile(cfg_file)

    estimator = build_flow_estimator(cfg.model).cuda()

    imgs = torch.randn(1, 6, 64, 64).cuda()
    flow_gt = torch.randn(1, 2, 64, 64).cuda()
    valid = torch.ones((1, 64, 64)).cuda()
    # test forward_test out
    assert isinstance(estimator(imgs, test_mode=True), list)

    losses = estimator(imgs, flow_gt, valid, test_mode=False)
    loss, _ = estimator._parse_losses(losses)

    assert float(loss.item()) > 0


@pytest.mark.parametrize('cfg_file', [
    '../../configs/_base_/models/raft.py',
    '../../configs/_base_/models/flownets.py',
    '../../configs/_base_/models/flownet2/flownet2sd.py',
    '../../configs/_base_/models/gma/gma.py',
    '../../configs/_base_/models/gma/gma_p-only.py',
    '../../configs/_base_/models/gma/gma_plus-p.py'
])
def test_flow_estimator_without_cuda(cfg_file):
    # BaseFlowEstimator has abstract method
    with pytest.raises(TypeError):
        FlowEstimator(init_cfg=None)

    cfg_file = osp.join(osp.dirname(__file__), cfg_file)
    cfg = Config.fromfile(cfg_file)
    if cfg.model.type == 'RAFT':
        # Replace SyncBN with BN to inference on CPU
        cfg.model.cxt_encoder.norm_cfg = dict(type='BN', requires_grad=True)

    estimator = build_flow_estimator(cfg.model)
    imgs = torch.randn(1, 6, 64, 64)
    flow_gt = torch.randn(1, 2, 64, 64)
    valid = torch.ones((1, 64, 64))
    # test forward_test out
    assert isinstance(estimator(imgs, test_mode=True), list)

    losses = estimator(imgs, flow_gt, valid, test_mode=False)
    loss, _ = estimator._parse_losses(losses)

    assert float(loss.item()) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_irr_pwc():

    cfg_file = '../../configs/_base_/models/irrpwc.py'
    cfg_file = osp.join(osp.dirname(__file__), cfg_file)
    cfg = Config.fromfile(cfg_file)

    estimator = build_flow_estimator(cfg.model).cuda()

    imgs = torch.randn(1, 6, 64, 64).cuda()
    flow_fw_gt = torch.randn(1, 2, 64, 64).cuda()
    flow_bw_gt = torch.randn(1, 2, 64, 64).cuda()
    occ_fw_gt = torch.randn(1, 1, 64, 64).cuda()
    occ_bw_gt = torch.randn(1, 1, 64, 64).cuda()

    # test forward_test out
    assert isinstance(estimator(imgs, test_mode=True), list)

    # test forward_train out with flow_fw_gt, flow_bw_gt, occ_fw_gt, occ_bw_gt
    losses = estimator(
        imgs,
        flow_fw_gt=flow_fw_gt,
        flow_bw_gt=flow_bw_gt,
        occ_fw_gt=occ_fw_gt,
        occ_bw_gt=occ_bw_gt,
        test_mode=False)
    loss, _ = estimator._parse_losses(losses)

    assert float(loss.item()) > 0

    # test forward_train out with flow_gt
    losses = estimator(
        imgs,
        flow_fw_gt=None,
        flow_bw_gt=None,
        occ_fw_gt=None,
        occ_bw_gt=None,
        flow_gt=flow_fw_gt,
        test_mode=False)
    loss, _ = estimator._parse_losses(losses)

    assert float(loss.item()) > 0

    # test forward_train out with flow_fw_gt, flow_bw_gt
    losses = estimator(
        imgs, flow_fw_gt=flow_fw_gt, flow_bw_gt=flow_bw_gt, test_mode=False)
    loss, _ = estimator._parse_losses(losses)

    # test forward_train out with flow_gt, occ_gt
    losses = estimator(
        imgs, flow_gt=flow_fw_gt, occ_gt=occ_fw_gt, test_mode=False)
    loss, _ = estimator._parse_losses(losses)
