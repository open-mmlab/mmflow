# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest

from mmflow.datasets.pipelines.advanced_transform import (RandomAffine,
                                                          check_out_of_bound,
                                                          theta_is_valid,
                                                          transform_flow)
from mmflow.datasets.utils import read_flow

img1_ = '../data/0000000-img_0.png'
img2_ = '../data/0000000-img_1.png'
flow_fw_ = '../data/0000000-flow_01.flo'
flow_bw_ = '../data/0000000-flow_10.flo'
occ_fw_ = '../data/0000000-occ_01.png'
occ_bw_ = '../data/0000000-occ_10.png'


def test_is_valid():

    theta_valid = np.array([[1.1, 0., 0], [0., 1.1, 0], [0., 0., 1.]])

    assert theta_is_valid(theta_valid)

    theta_invalid = np.array([[0.5, -np.sqrt(3) / 2., 0.2],
                              [np.sqrt(3) / 2., 0.5, 0.3], [0., 0., 1.]])

    assert not theta_is_valid(theta_invalid)


def test_transform_flow():

    h, w = 384, 512

    flow = np.random.randn(h, w, 2)
    theta1 = np.random.randn(3, 3)
    theta1[2, :2] = 0.
    theta1[2, 2] = 1.

    theta2 = np.random.randn(3, 3)
    theta2[2, :2] = 0.
    theta2[2, 2] = 1.

    new_flow = transform_flow(flow, None, theta1, theta2, h, w)

    assert new_flow.shape == (h, w, 2)


def test_apply_random_affine_to_theta():

    transform = dict(
        translates=(0.1, 0.1),
        zoom=(1.0, 1.5),
        shear=(0.86, 1.16),
        rotate=(-10., 10.))

    random_affine = RandomAffine()

    theta = np.eye(3)
    new_theta = random_affine._apply_random_affine_to_theta(theta, **transform)

    assert theta_is_valid(new_theta)


def test_check_out_of_bound():

    flow = read_flow(osp.join(osp.dirname(__file__), flow_fw_))

    h, w, _ = flow.shape
    occ = (np.random.randn(h, w) < 0).astype(flow.dtype)

    new_occ = check_out_of_bound(flow, occ)

    assert new_occ.shape == occ.shape

    assert np.sum(new_occ == 0) + np.sum(new_occ == 1) == h * w


def test_RandomAffine():

    invalid_transform = dict(
        translate=(0.1, 0.1),
        zoom=(1.0, 1.5),
        shear=(0.86, 1.16),
        rotate=(-10., 10.))

    global_transform = dict(
        translates=(0.1, 0.1),
        zoom=(1.0, 1.5),
        shear=(0.86, 1.16),
        rotate=(-10., 10.))

    relative_transform = dict(
        translates=(0.0075, 0.0075),
        zoom=(0.985, 1.015),
        shear=(1., 1.),
        rotate=(-1., 1.))

    with pytest.raises(AssertionError):
        RandomAffine(invalid_transform)

    random_affine = RandomAffine(
        global_transform,
        relative_transform,
        preserve_valid=True,
        check_bound=False)

    img1 = mmcv.imread(osp.join(osp.dirname(__file__), img1_))
    img2 = mmcv.imread(osp.join(osp.dirname(__file__), img2_))

    flow = mmcv.flowread(osp.join(osp.dirname(__file__), flow_fw_))

    results = dict()
    results['img1'] = img1
    results['img2'] = img2
    results['flow_gt'] = flow

    results['img_shape'] = img1.shape
    results['img_fields'] = ['img1', 'img2']
    results['ann_fields'] = ['flow_gt']

    new_results = random_affine(results)

    assert new_results
