# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from mmflow.models.losses import (MultiLevelBCE, MultiLevelCharbonnierLoss,
                                  MultiLevelEPE, SequenceLoss,
                                  multi_levels_binary_cross_entropy,
                                  sequence_loss, smooth_1st_loss,
                                  smooth_2nd_loss, weighted_ssim)
from mmflow.models.losses.census_loss import census_loss
from mmflow.models.losses.multilevel_charbonnier_loss import charbonnier_loss
from mmflow.models.losses.multilevel_epe import endpoint_error
from mmflow.models.losses.multilevel_flow_loss import multi_level_flow_loss


def test_multi_level_endpoint_error():

    b, h, w = 1, 4, 4

    pred = dict(level1=torch.randn(b, 2, h, w))
    gt = torch.randn(b, 2, h, w)
    weights = dict(level1=1.)

    # test gt channels is not 2
    with pytest.raises(AssertionError):
        multi_level_flow_loss(
            endpoint_error, pred, torch.randn(b, 1, 1, 1), weights=weights)

    # test invalid weight type
    with pytest.raises(AssertionError):
        multi_level_flow_loss(endpoint_error, pred, gt, p=3, weights=None)

    p = 2
    epe_gt = torch.mean(torch.norm(pred['level1'] - gt, p, dim=1))
    epe = multi_level_flow_loss(
        endpoint_error, pred, gt, weights, p=2, reduction='mean', flow_div=1.)

    assert torch.allclose(epe, epe_gt)

    valid = torch.zeros((b, h, w))
    valid[0, ...].fill_diagonal_(1, wrap=True)

    epe_gt = torch.sum(torch.norm(
        (pred['level1'] - gt), dim=1, p=p) * valid) / torch.sum(valid)
    epe = multi_level_flow_loss(
        endpoint_error,
        pred,
        gt,
        weights,
        valid=valid,
        p=2,
        reduction='mean',
        flow_div=1.)

    assert torch.allclose(epe, epe_gt)


def test_multi_level_binary_cross_entropy():

    b, h, w = 2, 64, 64

    occ_pred = dict(level1=torch.rand(b, 1, h, w))
    occ_gt = (torch.randint(low=0, high=2, size=(b, 1, h, w))).float()
    weights = dict(level1=1.)

    # test pred does not match gt
    with pytest.raises(AssertionError):
        multi_levels_binary_cross_entropy(
            occ_pred, torch.randn(b, 2, 1, 1), weights=weights)
    loss = multi_levels_binary_cross_entropy(
        occ_pred, occ_gt, balance=True, reduction='sum', weights=weights)

    # implementation from IRR released code
    def answer(y_pred, y_true):
        eps = 1e-8

        y_pred = torch.sigmoid(y_pred)

        tp = -(y_true *
               torch.log(y_pred + eps)).sum(dim=2).sum(dim=2).sum(dim=1)
        fn = -((1 - y_true) *
               torch.log((1 - y_pred) + eps)).sum(dim=2).sum(dim=2).sum(dim=1)

        denom_tp = y_true.sum(dim=2).sum(dim=2).sum(dim=1) + y_pred.sum(
            dim=2).sum(dim=2).sum(dim=1) + eps
        denom_fn = (1 - y_true).sum(dim=2).sum(dim=2).sum(
            dim=1) + (1 - y_pred).sum(dim=2).sum(dim=2).sum(dim=1) + eps

        return (
            (tp / denom_tp).sum() +
            (fn / denom_fn).sum()) * y_pred.size(2) * y_pred.size(3) * 0.5 / b

    assert torch.allclose(loss, answer(occ_pred['level1'], occ_gt))


def test_multi_level_charbonnier_loss():
    b, h, w = 1, 4, 4

    pred = dict(level1=torch.randn(b, 2, h, w))
    gt = torch.randn(b, 2, h, w)
    weights = dict(level1=1.)

    # test gt channels is not 2
    with pytest.raises(AssertionError):
        multi_level_flow_loss(
            charbonnier_loss, pred, torch.randn(b, 1, 1, 1), weights=weights)

    # test invalid weight type
    with pytest.raises(AssertionError):
        multi_level_flow_loss(charbonnier_loss, pred, gt, weights=None)

    q = 0.2
    eps = 0.01
    loss_gt = torch.mean((torch.sum((pred['level1'] - gt)**2, dim=1) + eps)**q)

    loss = multi_level_flow_loss(
        charbonnier_loss,
        pred,
        gt,
        weights=weights,
        q=q,
        eps=eps,
        reduction='mean',
        flow_div=1.)
    assert torch.allclose(loss_gt, loss)


@pytest.mark.parametrize('reduction', ('mean', 'sum'))
@pytest.mark.parametrize('resize_flow', ['upsample', 'downsample'])
@pytest.mark.parametrize('scale_as_level', [True, False])
def test_multilevel_epe(reduction, resize_flow, scale_as_level):

    b = 8

    flow2 = torch.randn(b, 2, 12, 16)
    flow3 = torch.randn(b, 2, 6, 8)

    gt = torch.randn(b, 2, 24, 32)

    preds_list = [flow2, flow3]
    preds = {
        'level2': flow2,
        'level3': flow3,
    }
    weights = {'level2': 0.005, 'level3': 0.01}

    with pytest.raises(AssertionError):
        MultiLevelEPE(flow_div=-1)

    with pytest.raises(AssertionError):
        MultiLevelEPE(weights=[0.005, 0.01])

    # test reduction value
    with pytest.raises(AssertionError):
        MultiLevelEPE(reduction=None)

    # test invalid resize_flow
    with pytest.raises(AssertionError):
        MultiLevelEPE(resize_flow='z')

    # test invalid scale_as_level
    with pytest.raises(AssertionError):
        MultiLevelEPE(scale_as_level='a')

    def answer():
        loss = 0
        weights = [0.005, 0.01]
        scales = [2, 4]
        scale_factor = [1 / 2, 1 / 4] if scale_as_level else [1., 1.]
        div_gt = gt / 20.

        for i in range(len(weights)):
            if resize_flow == 'downsample':
                cur_gt = F.avg_pool2d(div_gt, scales[i]) * scale_factor[i]
                cur_pred = preds_list[i]
            else:
                cur_gt = div_gt * scale_factor[i]
                cur_pred = F.interpolate(
                    preds_list[i],
                    size=(24, 32),
                    mode='bilinear',
                    align_corners=False)

            l2_loss = torch.norm(cur_pred - cur_gt, p=2, dim=1)
            if reduction == 'mean':
                loss += l2_loss.mean() * weights[i]
            else:
                loss += l2_loss.sum() / b * weights[i]

        return loss

    answer_ = answer()

    # test accuracy of mean reduction
    loss_func = MultiLevelEPE(
        weights=weights,
        reduction=reduction,
        resize_flow=resize_flow,
        scale_as_level=scale_as_level)
    loss = loss_func(preds, gt)
    assert torch.isclose(answer_, loss, atol=1e-4)

    valid = torch.zeros_like(gt[:, 0, :, :])
    loss = loss_func(preds, gt, valid)
    assert torch.isclose(torch.Tensor([0.]), loss, atol=1e-4)


@pytest.mark.parametrize('reduction', ('mean', 'sum'))
def test_multilevel_bce(reduction):

    b = 8

    occ2 = torch.randn(b, 1, 16, 16)
    occ3 = torch.randn(b, 1, 8, 8)

    gt = torch.randn(b, 1, 64, 64)

    preds_list = [occ2, occ3]
    preds = {
        'level2': occ2,
        'level3': occ3,
    }

    with pytest.raises(AssertionError):
        MultiLevelBCE(balance=1)

    with pytest.raises(AssertionError):
        MultiLevelBCE(weights=[0.005])

    loss_obj = MultiLevelBCE(
        balance=True,
        weights={
            'level2': 0.005,
            'level3': 0.01,
        },
        reduction=reduction)

    loss = loss_obj(preds, gt)

    def answer():

        def _single(y_pred, y_true):
            eps = 1e-8
            h = y_pred.size(2)
            w = y_pred.size(3)

            pred = torch.sigmoid(y_pred)

            tp = -(y_true * torch.log(pred + eps)).sum(dim=(1))
            fn = -((1 - y_true) * torch.log((1 - pred) + eps)).sum(dim=(1))

            denom_tp = y_true.sum(dim=(1, 2, 3)) + pred.sum(dim=(1, 2,
                                                                 3)) + eps
            denom_fn = (1 - y_true).sum(dim=(1, 2, 3)) + (1 - pred).sum(
                dim=(1, 2, 3)) + eps
            if reduction == 'mean':
                return (((tp / denom_tp.view(b, 1, 1)) +
                         (fn / denom_fn.view(b, 1, 1))) * h * w * 0.5).mean()
            else:
                return ((tp / denom_tp.view(b, 1, 1)).sum() +
                        (fn / denom_fn.view(b, 1, 1)).sum()) * h * w * 0.5 / b

        loss = 0
        weights = [0.005, 0.01]
        scales = [4, 8]

        for i in range(len(weights)):
            cur_gt = F.avg_pool2d(gt, scales[i])
            loss += _single(preds_list[i], cur_gt) * weights[i]
        return loss

    answer_ = answer()

    assert torch.isclose(loss, answer_, atol=1e-4)


def test_sequence_loss():
    pred = []
    flow_gt = torch.ones((1, 2, 10, 10))
    for _ in range(2):
        pred.append(torch.zeros((1, 2, 10, 10)))

    flow_loss = sequence_loss(
        preds=pred,
        flow_gt=flow_gt,
        gamma=1.,
        valid=torch.ones((1, 1, 10, 10)))

    assert flow_loss == 2.

    sequence_loss_module = SequenceLoss(gamma=1.)
    assert sequence_loss_module(
        pred, flow_gt, valid=torch.ones((1, 1, 10, 10)))


@pytest.mark.parametrize('reduction', ('mean', 'sum'))
@pytest.mark.parametrize('resize_flow', ['upsample', 'downsample'])
@pytest.mark.parametrize('scale_as_level', [True, False])
def test_multi_levels_charbonnier(reduction, resize_flow, scale_as_level):

    b = 2

    flow2 = torch.randn(b, 2, 12, 16)
    flow3 = torch.randn(b, 2, 6, 8)

    gt = torch.randn(b, 2, 24, 32)

    preds_list = [flow2, flow3]
    preds = {
        'level2': flow2,
        'level3': flow3,
    }
    weights = {'level2': 0.005, 'level3': 0.01}

    with pytest.raises(AssertionError):
        MultiLevelCharbonnierLoss(flow_div=-1)

    with pytest.raises(AssertionError):
        MultiLevelCharbonnierLoss(weights=[0.005, 0.01])

    # test reduction value
    with pytest.raises(AssertionError):
        MultiLevelEPE(reduction=None)

    # test invalid resize_flow
    with pytest.raises(AssertionError):
        MultiLevelEPE(resize_flow='z')

    # test invalid scale_as_level
    with pytest.raises(AssertionError):
        MultiLevelEPE(scale_as_level='a')

    def answer():
        loss = 0
        weights = [0.005, 0.01]
        scales = [2, 4]
        scale_factor = [1 / 2, 1 / 4] if scale_as_level else [1., 1.]
        div_gt = gt / 20.

        for i in range(len(weights)):

            if resize_flow == 'downsample':
                cur_gt = F.avg_pool2d(div_gt, scales[i]) * scale_factor[i]
                cur_pred = preds_list[i]
            else:
                cur_gt = div_gt * scale_factor[i]
                cur_pred = F.interpolate(
                    preds_list[i],
                    size=(24, 32),
                    mode='bilinear',
                    align_corners=False)

            loss_square = torch.sum((cur_pred - cur_gt)**2, dim=1)
            if reduction == 'mean':
                loss += ((loss_square + 0.01)**0.2).mean() * weights[i]
            else:
                loss += ((loss_square + 0.01)**0.2).sum() / b * weights[i]

        return loss

    answer_ = answer()

    # test accuracy of mean reduction
    loss_obj = MultiLevelCharbonnierLoss(
        weights=weights, resize_flow=resize_flow, reduction=reduction)
    loss = loss_obj(preds, gt)
    assert torch.isclose(answer_, loss, rtol=1e-2)

    valid = torch.zeros_like(gt[:, 0, :, :])
    loss = loss_obj(preds, gt, valid)
    assert torch.isclose(torch.Tensor([0.]), loss, rtol=1e-2)


@pytest.mark.parametrize('kwargs', [
    dict(c1=0.01**2, c2=0.03**2),
    dict(c1=float('inf'), c2=0.03**2),
    dict(c1=0.01**2, c2=float('inf'))
])
def test_weighted_ssim(kwargs):
    B, C, H, W = 4, 2, 7, 7
    x = torch.ones((B, C, H, W), dtype=torch.float32)
    y_like = x * 2.
    y_unlike = x * torch.randn((H, W))

    # test c1 c2 input value
    with pytest.raises(ValueError):
        c1 = float('inf')
        c2 = float('inf')
        weighted_ssim(x, y_like, c1=c1, c2=c2)

    # test weight with invalid shape
    with pytest.raises(AssertionError):
        weighted_ssim(x, y_like, weight=torch.randn((1, 2)))

    ssim_like = weighted_ssim(x, y_like, **kwargs)
    ssim_unlike = weighted_ssim(x, y_unlike, **kwargs)

    assert ssim_unlike.shape == ssim_like.shape == (B, C, H - 2, W - 2)

    assert ssim_like.mean() < ssim_unlike.mean()


@pytest.mark.parametrize('smooth_edge_weighting', ('exponential', 'gaussian'))
@pytest.mark.parametrize('loss_func', (smooth_1st_loss, smooth_2nd_loss))
def test_smooth_loss(smooth_edge_weighting, loss_func):
    B, H, W = (1, 5, 5)

    image_smooth = torch.ones((B, 3, H, W)) + torch.randn((B, 3, H, W)) * 0.1
    image_unsmooth = torch.ones(B, 3, H, W) + torch.randn((B, 3, H, W)) * 5

    flow_smooth = torch.ones((B, 2, H, W)) * 2 + torch.randn(
        (B, 2, H, W)) * 0.1
    flow_unsmooth = torch.ones((B, 2, H, W)) * 2 + torch.randn(
        (B, 2, H, W)) * 5

    # test invalid smoothness_edge_weighting
    with pytest.raises(AssertionError):
        loss_func(flow_smooth, image_smooth, smooth_edge_weighting='a')

    # test alpha = 0 that image will not effect loss
    assert loss_func(
        flow_smooth,
        image_unsmooth,
        alpha=0,
        smooth_edge_weighting=smooth_edge_weighting) == loss_func(
            flow_smooth,
            image_smooth,
            alpha=0,
            smooth_edge_weighting=smooth_edge_weighting)
    assert loss_func(
        flow_unsmooth,
        image_unsmooth,
        alpha=0,
        smooth_edge_weighting=smooth_edge_weighting) == loss_func(
            flow_unsmooth,
            image_smooth,
            alpha=0,
            smooth_edge_weighting=smooth_edge_weighting)

    # test image edge effect loss
    # flow is unsmooth but image is smooth so loss will bigger
    assert loss_func(
        flow_unsmooth,
        image_smooth,
        alpha=1,
        smooth_edge_weighting=smooth_edge_weighting) > loss_func(
            flow_unsmooth,
            image_unsmooth,
            alpha=1,
            smooth_edge_weighting=smooth_edge_weighting)


def test_census_loss():
    image1 = torch.randn(1, 3, 5, 5)
    image2 = torch.randn(1, 3, 5, 5)

    loss1 = census_loss(image1=image1, image2=image2)

    image2.data = image1.data

    loss2 = census_loss(image1=image1, image2=image2)

    assert loss2 == 0. and loss1.abs() > loss2
