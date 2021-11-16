# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmflow.core.evaluation import (end_point_error, end_point_error_map,
                                    eval_metrics, optical_flow_outliers)


def test_end_point_error_map():
    flow_pred = np.random.normal(size=(10, 10, 2))
    flow_gt = np.random.normal(size=(10, 10, 2))

    target = np.linalg.norm((flow_pred - flow_gt), ord=2, axis=-1)
    assert (target == end_point_error_map(flow_pred, flow_gt)).all()


def test_end_point_error():
    # test length of predicted flow map is not equal to length of gt map
    with pytest.raises(AssertionError):
        flow_pred = [np.random.normal(size=(10, 10, 2))] * 2
        flow_gt = [np.random.normal(size=(10, 10, 2))] * 3
        valid_gt = [np.ones((10, 10))] * 3
        end_point_error(flow_pred, flow_gt, valid_gt)

    flow_pred = [np.random.normal(size=(3, 3, 2))]
    flow_gt = [np.random.normal(size=(3, 3, 2))]
    valid_gt = [np.array([[1., 1., 1.], [0., 0., 0.], [1., 0.2, 0.3]])]

    target = np.linalg.norm((flow_pred[0] - flow_gt[0]), ord=2, axis=-1)
    target = (target[0].sum() + target[2][0]) / 4

    assert target == end_point_error(flow_pred, flow_gt, valid_gt)


def test_optical_flow_outliers():
    # test length of predicted flow map is not equal to length of gt map
    with pytest.raises(AssertionError):
        flow_pred = [np.random.normal(size=(10, 10, 2))] * 2
        flow_gt = [np.random.normal(size=(10, 10, 2))] * 3
        valid_gt = [np.ones((10, 10))] * 3
        optical_flow_outliers(flow_pred, flow_gt, valid_gt)

    flow_pred = [np.array([[[10., 5.], [0.1, 3.]], [[3., 15.2], [2.4, 4.5]]])]
    flow_gt = [np.array([[[10.1, 4.8], [10, 3.]], [[6., 10.2], [2.0, 4.1]]])]
    valid_gt = [np.array([[1., 1.], [1., 0.3]])]

    target = 100 * (2 / 3)
    assert target == optical_flow_outliers(flow_pred, flow_gt, valid_gt)


def test_eval_metrics():
    # test valid metric types
    with pytest.raises(KeyError):
        flow_pred = np.random.normal(size=(10, 10, 2))
        flow_gt = np.random.normal(size=(10, 10, 2))
        valid_gt = np.ones((10, 10))
        eval_metrics(flow_pred, flow_gt, valid_gt, metrics=['abc'])

    flow_pred = [np.array([[[10., 5.], [0.1, 3.]], [[3., 15.2], [2.4, 4.5]]])]
    flow_gt = [np.array([[[10.1, 4.8], [10, 3.]], [[6., 10.2], [2.0, 4.1]]])]
    valid_gt = [np.array([[1., 1.], [1., 0.3]])]
    result_dict = eval_metrics(
        flow_pred, flow_gt, valid_gt, metrics=['EPE', 'Fl'])

    epe_tar = np.linalg.norm((flow_pred[0] - flow_gt[0]), ord=2, axis=-1)
    epe_tar = (epe_tar[0].sum() + epe_tar[1][0]) / 3
    fl_tar = 100 * (2 / 3)

    assert epe_tar == result_dict['EPE']
    assert fl_tar == result_dict['Fl']
