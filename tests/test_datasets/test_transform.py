# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import cv2
import mmcv
import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmflow.datasets import Compose
from mmflow.datasets.builder import PIPELINES
from mmflow.datasets.utils import read_flow

img1_ = '../data/0000000-img_0.png'
img2_ = '../data/0000000-img_1.png'
flow_fw_ = '../data/0000000-flow_01.flo'
flow_bw_ = '../data/0000000-flow_10.flo'
occ_fw_ = '../data/0000000-occ_01.png'
occ_bw_ = '../data/0000000-occ_10.png'


def make_testdata_for_train():
    results = dict()
    img1 = mmcv.imread(osp.join(osp.dirname(__file__), img1_), 'color')
    original_img1 = copy.deepcopy(img1)
    img2 = mmcv.imread(osp.join(osp.dirname(__file__), img2_), 'color')
    original_img2 = copy.deepcopy(img2)
    flow_fw = read_flow(osp.join(osp.dirname(__file__), flow_fw_))
    original_flow_fw = copy.deepcopy(flow_fw)
    flow_bw = read_flow(osp.join(osp.dirname(__file__), flow_bw_))
    original_flow_bw = copy.deepcopy(flow_bw)
    occ_fw = mmcv.imread(
        osp.join(osp.dirname(__file__), occ_fw_), flag='grayscale')
    original_occ_fw = copy.deepcopy(occ_fw)
    occ_bw = mmcv.imread(
        osp.join(osp.dirname(__file__), occ_bw_), flag='grayscale')
    original_occ_bw = copy.deepcopy(occ_bw)

    results['img1'] = img1
    results['img2'] = img2
    results['flow_fw_gt'] = flow_fw
    results['flow_bw_gt'] = flow_bw
    results['occ_fw_gt'] = occ_fw
    results['occ_bw_gt'] = occ_bw
    results['img_fields'] = ['img1', 'img2']
    results['ann_fields'] = [
        'flow_fw_gt', 'flow_bw_gt', 'occ_fw_gt', 'occ_bw_gt'
    ]
    results['img_shape'] = img1.shape
    results['ori_shape'] = img1.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img1.shape
    results['scale_factor'] = 1.0
    return results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw


def make_testdata_for_test():
    results = dict()
    img1 = mmcv.imread(osp.join(osp.dirname(__file__), img1_), 'color')
    original_img1 = copy.deepcopy(img1)
    img2 = mmcv.imread(osp.join(osp.dirname(__file__), img2_), 'color')
    original_img2 = copy.deepcopy(img2)
    results['img1'] = img1
    results['img2'] = img2
    results['img_fields'] = ['img1', 'img2']
    results['img_shape'] = img1.shape
    results['ori_shape'] = img1.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img1.shape
    results['scale_factor'] = 1.0
    return results, original_img1, original_img2


def test_flip():
    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', prob=1.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', prob=1, direction='horizonta')
        build_from_cfg(transform, PIPELINES)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()

    transform = dict(type='RandomFlip', prob=1)
    flip_module = build_from_cfg(transform, PIPELINES)
    results = flip_module(results)
    assert np.equal(original_img1[:, ::-1, :], results['img1']).all()
    assert np.equal(original_img2[:, ::-1, :], results['img2']).all()
    assert np.equal(original_flow_fw[:, ::-1, :] * [-1, 1],
                    results['flow_fw_gt']).all()
    assert np.equal(original_flow_bw[:, ::-1, :] * [-1, 1],
                    results['flow_bw_gt']).all()
    assert np.equal(original_occ_fw[:, ::-1], results['occ_fw_gt']).all()
    assert np.equal(original_occ_bw[:, ::-1], results['occ_bw_gt']).all()
    assert results['flip'] == [True]
    assert results['flip_direction'] == ['horizontal']

    results = flip_module(results)
    assert np.equal(original_img1, results['img1']).all()
    assert np.equal(original_img2, results['img2']).all()
    assert np.equal(original_flow_fw, results['flow_fw_gt']).all()
    assert np.equal(original_flow_bw, results['flow_bw_gt']).all()
    assert np.equal(original_occ_fw, results['occ_fw_gt']).all()
    assert np.equal(original_occ_bw, results['occ_bw_gt']).all()
    assert results['flip'] == [True, True]
    assert results['flip_direction'] == ['horizontal', 'horizontal']

    results, original_img1, original_img2 = make_testdata_for_test()
    results = flip_module(results)
    assert np.equal(original_img1[:, ::-1, :], results['img1']).all()
    assert np.equal(original_img2[:, ::-1, :], results['img2']).all()
    results = flip_module(results)
    assert np.equal(original_img1, results['img1']).all()
    assert np.equal(original_img2, results['img2']).all()


def test_random_crop():
    # test assertion for invalid random crop
    with pytest.raises(AssertionError):
        transform = dict(type='RandomCrop', crop_size=(-1, 0))
        build_from_cfg(transform, PIPELINES)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()

    h, w, _ = original_img1.shape
    transform = dict(type='RandomCrop', crop_size=(h - 20, w - 20))
    crop_module = build_from_cfg(transform, PIPELINES)
    results = crop_module(results)
    assert results['img1'].shape == (h - 20, w - 20, 3)
    assert results['img_shape'] == (h - 20, w - 20, 3)
    assert results['flow_fw_gt'].shape == (h - 20, w - 20, 2)
    assert results['flow_bw_gt'].shape == (h - 20, w - 20, 2)
    assert results['occ_fw_gt'].shape == (h - 20, w - 20)
    assert results['occ_fw_gt'].shape == (h - 20, w - 20)
    crop_y1, crop_y2, crop_x1, crop_x2 = results['crop_bbox']
    assert np.all(results['img1'] == original_img1[crop_y1:crop_y2,
                                                   crop_x1:crop_x2, ...])
    assert np.all(results['img2'] == original_img2[crop_y1:crop_y2,
                                                   crop_x1:crop_x2, ...])
    assert np.all(results['flow_fw_gt'] == original_flow_fw[crop_y1:crop_y2,
                                                            crop_x1:crop_x2,
                                                            ...])
    assert np.all(results['flow_bw_gt'] == original_flow_bw[crop_y1:crop_y2,
                                                            crop_x1:crop_x2,
                                                            ...])
    assert np.all(results['occ_fw_gt'] == original_occ_fw[crop_y1:crop_y2,
                                                          crop_x1:crop_x2,
                                                          ...])
    assert np.all(results['occ_bw_gt'] == original_occ_bw[crop_y1:crop_y2,
                                                          crop_x1:crop_x2,
                                                          ...])
    results, original_img1, original_img2 = make_testdata_for_test()
    results = crop_module(results)
    assert results['img1'].shape == (h - 20, w - 20, 3)
    assert results['img_shape'] == (h - 20, w - 20, 3)
    crop_y1, crop_y2, crop_x1, crop_x2 = results['crop_bbox']
    assert np.all(results['img1'] == original_img1[crop_y1:crop_y2,
                                                   crop_x1:crop_x2, ...])
    assert np.all(results['img2'] == original_img2[crop_y1:crop_y2,
                                                   crop_x1:crop_x2, ...])


@pytest.mark.parametrize('max_flow', (-1, 5., 1e5))
def test_validation(max_flow):
    # test assertion for invalid max_flow
    with pytest.raises(AssertionError):
        transform = dict(type='Validation', max_flow='10')
        build_from_cfg(transform, PIPELINES)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()

    transform = dict(type='Validation', max_flow=max_flow)
    val_module = build_from_cfg(transform, PIPELINES)
    results = val_module(results)
    assert results['valid_fw'].shape == original_flow_fw.shape[:2]
    assert results['valid_bw'].shape == original_flow_fw.shape[:2]
    assert np.all(results['img1'] == original_img1)
    assert np.all(results['img2'] == original_img2)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)

    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)
    if max_flow == -1:
        assert results['max_flow'] == -1
        assert results['valid_fw'].sum() == 0
        assert results['valid_bw'].sum() == 0
    elif max_flow == 1e5:
        assert results['max_flow'] == 1e5
        assert results['valid_fw'].sum() == np.prod(original_flow_fw.shape[:2])
        assert results['valid_bw'].sum() == np.prod(original_flow_fw.shape[:2])
    elif max_flow == 5:
        assert results['max_flow'] == 5
        assert results['valid_fw'].sum() == 2493
        assert results['valid_bw'].sum() == 2500


@pytest.mark.parametrize('max_num', [2, 10])
def test_erase(max_num):

    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='Erase', prob='1')
        build_from_cfg(transform, PIPELINES)
        transform = dict(type='Erase', prob=-1)
        build_from_cfg(transform, PIPELINES)

    # test invalid erase
    with pytest.raises(AssertionError):
        transform = dict(type='Erase', prob=1., max_num=1.4)
        build_from_cfg(transform, PIPELINES)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()

    transform = dict(type='Erase', prob=1., max_num=max_num)
    erase_module = build_from_cfg(transform, PIPELINES)
    results = erase_module(results)
    assert results['erase_num'] == len(results['erase_bounds'])
    num_pixels = 0
    for box in results['erase_bounds']:
        num_pixels += (box[2] - box[0]) * (box[3] - box[1])
    assert np.sum(original_img2 != results['img2']) <= num_pixels * 3
    assert np.all(results['img1'] == original_img1)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)

    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)

    results, original_img1, original_img2 = make_testdata_for_test()
    results = erase_module(results)
    assert results['erase_num'] == len(results['erase_bounds'])
    num_pixels = 0
    for box in results['erase_bounds']:
        num_pixels += (box[2] - box[0]) * (box[3] - box[1])
    assert np.sum(original_img2 != results['img2']) <= num_pixels * 3
    assert np.all(results['img1'] == original_img1)


def test_input_resize():
    with pytest.raises(AssertionError):
        # test invalid exponent
        transform = dict(type='InputResize', exponent=3.2)
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='InputResize', exponent=9)
    resize_module = build_from_cfg(transform, PIPELINES)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()

    results = resize_module(results)
    assert results['img_shape'][0] % 2**9 == 0
    assert results['img_shape'][1] % 2**9 == 0

    assert np.all(
        results['scale_factor'] == np.array([10.24, 10.24], dtype=np.float32))
    assert np.all(results['img1'] == cv2.resize(
        original_img1, dsize=results['img_shape'][:2]))
    assert np.all(results['img2'] == cv2.resize(
        original_img2, dsize=results['img_shape'][:2]))

    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)

    results, original_img1, original_img2 = make_testdata_for_test()
    results = resize_module(results)
    assert results['img_shape'][0] % 2**9 == 0
    assert results['img_shape'][1] % 2**9 == 0
    assert np.all(
        results['scale_factor'] == np.array([10.24, 10.24], dtype=np.float32))
    assert np.all(results['img1'] == cv2.resize(
        original_img1, dsize=results['img_shape'][:2]))
    assert np.all(results['img2'] == cv2.resize(
        original_img2, dsize=results['img_shape'][:2]))


def test_input_pad():
    with pytest.raises(AssertionError):
        # test invalid exponent
        transform = dict(type='InputResize', exponent=3.2)
        build_from_cfg(transform, PIPELINES)
        # test invalid position
        transform = dict(type='InputPad', exponent=2, position=None)
        build_from_cfg(transform, PIPELINES)

    transform = dict(
        type='InputPad',
        exponent=9,
        position='center',
        mode='constant',
        constant_values=(0))

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()
    pad_module = build_from_cfg(transform, PIPELINES)
    results = pad_module(results)
    assert results['pad_shape'][0] % 2**9 == 0
    assert results['pad_shape'][1] % 2**9 == 0
    H, W = results['pad_shape'][:2]
    origin_H, origin_W = original_img1.shape[:2]
    pad_H = H - origin_H
    pad_W = W - origin_W

    assert np.all(results['img1'][pad_H // 2:H - pad_H // 2,
                                  pad_W // 2:W - pad_W // 2,
                                  ...] == original_img1)
    assert np.all(results['img2'][pad_H // 2:H - pad_H // 2,
                                  pad_W // 2:W - pad_W // 2,
                                  ...] == original_img2)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)

    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)

    results, original_img1, original_img2 = make_testdata_for_test()
    results = pad_module(results)
    assert results['pad_shape'][0] % 2**9 == 0
    assert results['pad_shape'][1] % 2**9 == 0
    H, W = results['pad_shape'][:2]
    origin_H, origin_W = original_img1.shape[:2]
    pad_H = H - origin_H
    pad_W = W - origin_W

    assert np.all(results['img1'][pad_H // 2:H - pad_H // 2,
                                  pad_W // 2:W - pad_W // 2,
                                  ...] == original_img1)
    assert np.all(results['img2'][pad_H // 2:H - pad_H // 2,
                                  pad_W // 2:W - pad_W // 2,
                                  ...] == original_img2)


def test_rgb2bgr():
    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()
    transform = dict(type='BGR2RGB')
    bgr2rgb_module = build_from_cfg(transform, PIPELINES)
    results = bgr2rgb_module(results)
    assert results['channels_order'] == 'RGB'
    assert np.all(results['img1'][:, :, 0] == original_img1[:, :, 2])
    assert np.all(results['img1'][:, :, 2] == original_img1[:, :, 0])
    assert np.all(results['img2'][:, :, 0] == original_img2[:, :, 2])
    assert np.all(results['img2'][:, :, 2] == original_img2[:, :, 0])
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)
    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)

    results, original_img1, original_img2 = make_testdata_for_test()
    results = bgr2rgb_module(results)
    assert results['channels_order'] == 'RGB'
    assert np.all(results['img1'][:, :, 0] == original_img1[:, :, 2])
    assert np.all(results['img1'][:, :, 2] == original_img1[:, :, 0])
    assert np.all(results['img2'][:, :, 0] == original_img2[:, :, 2])
    assert np.all(results['img2'][:, :, 2] == original_img2[:, :, 0])


def test_normalize():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=False)
    transform = dict(type='Normalize', **img_norm_cfg)
    transform = build_from_cfg(transform, PIPELINES)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()
    results = transform(results)

    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    converted_img1 = (original_img1.astype(np.float32) - mean) / std
    assert np.allclose(results['img1'], converted_img1)
    converted_img2 = (original_img2 - mean) / std
    assert np.allclose(results['img2'], converted_img2)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)
    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)

    results, original_img1, original_img2 = make_testdata_for_test()
    results = transform(results)
    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    converted_img1 = (original_img1.astype(np.float32) - mean) / std
    assert np.allclose(results['img1'], converted_img1)
    converted_img2 = (original_img2 - mean) / std
    assert np.allclose(results['img2'], converted_img2)


def test_rerange():
    # test assertion if min_value or max_value is illegal
    with pytest.raises(AssertionError):
        transform = dict(type='Rerange', min_value=[0], max_value=[255])
        build_from_cfg(transform, PIPELINES)

    # test assertion if min_value >= max_value
    with pytest.raises(AssertionError):
        transform = dict(type='Rerange', min_value=1, max_value=1)
        build_from_cfg(transform, PIPELINES)

    # test assertion if img_min_value == img_max_value
    with pytest.raises(AssertionError):
        transform = dict(type='Rerange', min_value=0, max_value=1)
        transform = build_from_cfg(transform, PIPELINES)
        results = dict()
        results['img1'] = np.array([[1, 1], [1, 1]])
        results['img2'] = np.array([[1, 1], [1, 1]])
        results['img_fields'] = ['img1', 'img2']
        transform(results)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()
    img_rerange_cfg = dict()
    transform = dict(type='Rerange', **img_rerange_cfg)
    transform = build_from_cfg(transform, PIPELINES)
    results = transform(results)

    min_value1 = np.min(original_img1)
    max_value1 = np.max(original_img1)
    converted_img1 = (original_img1 - min_value1) / (max_value1 -
                                                     min_value1) * 255
    assert np.allclose(results['img1'], converted_img1)
    min_value2 = np.min(original_img2)
    max_value2 = np.max(original_img2)
    converted_img2 = (original_img2 - min_value2) / (max_value2 -
                                                     min_value2) * 255
    assert np.allclose(results['img2'], converted_img2)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)
    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)

    assert str(transform) == f'Rerange(min_value={0}, max_value={255})'

    results, original_img1, original_img2 = make_testdata_for_test()
    results = transform(results)

    min_value1 = np.min(original_img1)
    max_value1 = np.max(original_img1)
    converted_img1 = (original_img1 - min_value1) / (max_value1 -
                                                     min_value1) * 255
    assert np.allclose(results['img1'], converted_img1)
    min_value2 = np.min(original_img2)
    max_value2 = np.max(original_img2)
    converted_img2 = (original_img2 - min_value2) / (max_value2 -
                                                     min_value2) * 255


@pytest.mark.parametrize('prob', [0., 1.])
def test_randomrotation(prob):
    # tets invalid prob angle
    with pytest.raises(AssertionError):
        transform = dict(type='RandomRotation', prob=1.2, angle=90.)
        build_from_cfg(transform, PIPELINES)
        transform = dict(type='RandomRotation', prob=0, angle=270.)
        build_from_cfg(transform, PIPELINES)

    # test prob = 0
    transform = dict(type='RandomRotation', prob=prob, angle=180.)
    transform = build_from_cfg(transform, PIPELINES)
    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()
    results = transform(results)
    if prob == 0:
        assert np.all(results['img1'] == original_img1)
        assert np.all(results['img2'] == original_img2)
        assert np.all(results['flow_fw_gt'] == original_flow_fw)
        assert np.all(results['flow_bw_gt'] == original_flow_bw)
        assert np.all(results['occ_fw_gt'] == original_occ_fw)
        assert np.all(results['occ_bw_gt'] == original_occ_bw)
        assert not results['rotate']
    # TODO: test rotated flow value
    if prob == 1.:
        assert np.all(results['img1'].shape == original_img1.shape)
        assert np.all(results['img2'].shape == original_img2.shape)
        assert np.all(results['flow_fw_gt'].shape == original_flow_fw.shape)
        assert np.all(results['flow_bw_gt'].shape == original_flow_bw.shape)
        assert np.all(results['occ_fw_gt'].shape == original_occ_fw.shape)
        assert np.all(results['occ_bw_gt'].shape == original_occ_bw.shape)
        assert results['rotate']
    results, original_img1, original_img2 = make_testdata_for_test()
    results = transform(results)
    if prob == 0:
        assert np.all(results['img1'] == original_img1)
        assert np.all(results['img2'] == original_img2)
    if prob == 1.:
        assert np.all(results['img1'].shape == original_img1.shape)
        assert np.all(results['img2'].shape == original_img2.shape)


def test_photometricdistortion():
    transform = dict(type='PhotoMetricDistortion')
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img1 = mmcv.imread(osp.join(osp.dirname(__file__), img1_), 'color')
    img2 = mmcv.imread(osp.join(osp.dirname(__file__), img1_), 'color')
    results['img1'] = img1
    results['img2'] = img2
    results['img_fields'] = ['img1', 'img2']
    # test Transform synchronization
    results = transform(results)
    assert np.all(results['img1'] == results['img2'])


def test_colorjitter():
    with pytest.raises(ValueError):
        # test arguments< 0
        transform = dict(
            type='ColorJitter',
            brightness=-1,
            contrast=0.,
            saturation=0.,
            hue=0.)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(ValueError):
        # test arguments range
        transform = dict(type='ColorJitter', hue=1.)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(TypeError):
        # test arguments type
        transform = dict(
            type='ColorJitter',
            brightness=0.,
            contrast=0.,
            saturation=0.,
            hue='0.')
        build_from_cfg(transform, PIPELINES)

    transform = dict(
        type='ColorJitter',
        brightness=0.1,
        contrast=0.3,
        saturation=0.1,
        hue=0.2)
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img1 = mmcv.imread(osp.join(osp.dirname(__file__), img1_), 'color')
    img2 = mmcv.imread(osp.join(osp.dirname(__file__), img1_), 'color')
    results['img1'] = img1
    results['img2'] = img2
    results['img_fields'] = ['img1', 'img2']
    # test Transform synchronization
    results = transform(results)
    assert np.all(results['img1'] == results['img2'])


def test_spacialtransform():
    # test invalid prob
    with pytest.raises(AssertionError):
        transform = dict(
            type='SpacialTransform',
            spacial_prob=1.1,
            stretch_prob=0.1,
            crop_size=(100, 100))
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(
            type='SpacialTransform',
            spacial_prob=0.1,
            stretch_prob=11.1,
            crop_size=(100, 100))
        build_from_cfg(transform, PIPELINES)
    # test invalid cropsize
    with pytest.raises(AssertionError):
        transform = dict(
            type='SpacialTransform',
            spacial_prob=0.1,
            stretch_prob=0.1,
            crop_size=1)
        build_from_cfg(transform, PIPELINES)
    # test invalid min_scale
    with pytest.raises(AssertionError):
        transform = dict(
            type='SpacialTransform',
            spacial_prob=0.1,
            stretch_prob=0.1,
            crop_size=(100, 100),
            min_scale='1.')
        build_from_cfg(transform, PIPELINES)
    # test invalid max_scale
    with pytest.raises(AssertionError):
        transform = dict(
            type='SpacialTransform',
            spacial_prob=0.1,
            stretch_prob=0.1,
            crop_size=(100, 100),
            max_scale='1.')
        build_from_cfg(transform, PIPELINES)
    # test invalid max_stretch
    with pytest.raises(AssertionError):
        transform = dict(
            type='SpacialTransform',
            spacial_prob=0.1,
            stretch_prob=0.1,
            crop_size=(100, 100),
            max_stretch='1.')
        build_from_cfg(transform, PIPELINES)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()
    # test spacial_prob = 0
    transform = dict(
        type='SpacialTransform',
        spacial_prob=0.0,
        stretch_prob=1.,
        crop_size=[100, 100],
    )
    transform = build_from_cfg(transform, PIPELINES)
    results = transform(results)

    assert np.all(results['img1'] == original_img1)
    assert np.all(results['img2'] == original_img2)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)
    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)
    assert results['scale'] == (1., 1.)

    # test spacial_prob = 1
    transform = dict(
        type='SpacialTransform',
        spacial_prob=1.,
        stretch_prob=0.,
        crop_size=[100, 120])
    transform = build_from_cfg(transform, PIPELINES)
    results = transform(results)

    assert results['img_shape'][:2] == (100, 120)
    assert results['img1'].shape == (100, 120, 3)
    assert results['img2'].shape == (100, 120, 3)
    assert results['flow_fw_gt'].shape == (100, 120, 2)
    assert results['flow_bw_gt'].shape == (100, 120, 2)
    assert results['occ_fw_gt'].shape == (100, 120)
    assert results['occ_bw_gt'].shape == (100, 120)


@pytest.mark.parametrize(('sigma_range', 'clamp_range'),
                         [((0., 0.04), (float('-inf'), float('inf'))),
                          ((0., 0.04), (0., 1.))])
def test_gaussiannoise(sigma_range, clamp_range):
    # test sigma_range type
    with pytest.raises(AssertionError):
        transform = dict(type='GaussianNoise', sigma_range=0.0)
        build_from_cfg(transform, PIPELINES)
    # test sigma_range < 0
    with pytest.raises(AssertionError):
        transform = dict(type='GaussianNoise', sigma_range=[-1, 1])
        build_from_cfg(transform, PIPELINES)
    # test sigma[0] > sigma[1]
    with pytest.raises(AssertionError):
        transform = dict(type='GaussianNoise', sigma_range=[2, 1])
        build_from_cfg(transform, PIPELINES)

    # test clamp_range type
    with pytest.raises(AssertionError):
        transform = dict(type='GaussianNoise', clamp_range=0.0)
        build_from_cfg(transform, PIPELINES)
    # test clamp[0] > clamp[1]
    with pytest.raises(AssertionError):
        transform = dict(type='GaussianNoise', clamp_range=[2, 1])
        build_from_cfg(transform, PIPELINES)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()

    # add gaussian noise on uint8 image
    with pytest.raises(AssertionError):
        transform = dict(type='GaussianNoise')
        transform = build_from_cfg(transform, PIPELINES)
        results = transform(results)

    img_norm_cfg = dict(mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True)

    transforms = [
        dict(type='Normalize', **img_norm_cfg),
        dict(
            type='GaussianNoise',
            sigma_range=sigma_range,
            clamp_range=clamp_range)
    ]
    transforms = Compose(transforms)
    results = transforms(results)
    assert np.all(results['img1'].shape == original_img1.shape)
    assert np.all(results['img2'].shape == original_img2.shape)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)
    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)
    assert sigma_range[0] <= results['sigma'] <= sigma_range[1]
    assert np.min(results['img1']) >= clamp_range[0]
    assert np.min(results['img2']) >= clamp_range[0]
    assert np.max(results['img1']) <= clamp_range[1]
    assert np.max(results['img2']) <= clamp_range[1]

    results, original_img1, original_img2 = make_testdata_for_test()
    results = transforms(results)
    assert np.all(results['img1'].shape == original_img1.shape)
    assert np.all(results['img2'].shape == original_img2.shape)


def test_translate():
    # test prob
    with pytest.raises(AssertionError):
        transform = dict(type='RandomTranslate', prob=-1)
        build_from_cfg(transform, PIPELINES)
    # test offset
    with pytest.raises(AssertionError):
        transform = dict(type='RandomTranslate', prob=0., x_offset=1.1)
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(TypeError):
        transform = dict(type='RandomTranslate', x_offset='st')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='RandomTranslate', prob=0.)
    transform = build_from_cfg(transform, PIPELINES)
    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()
    org_H, org_W = original_img1.shape[:2]
    results = transform(results)
    assert np.all(results['img1'] == original_img1)
    assert np.all(results['img2'] == original_img2)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)
    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)
    assert not results['translate']

    transform = dict(type='RandomTranslate', prob=1., x_offset=0., y_offset=0.)
    transform = build_from_cfg(transform, PIPELINES)
    results = transform(results)
    assert np.all(results['img1'] == original_img1)
    assert np.all(results['img2'] == original_img2)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)
    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)
    assert results['translate']
    transform = dict(
        type='RandomTranslate', prob=1., x_offset=0.05, y_offset=0.1)
    transform = build_from_cfg(transform, PIPELINES)
    results = transform(results)
    x_offset, y_offset = results['translate_offset']
    M = np.float32([[1, 0, x_offset * org_W], [0, 1, y_offset * org_H]])
    w = org_W
    h = org_H
    assert np.all(results['img1'] == cv2.warpAffine(original_img1, M, (w, h)))
    assert np.all(results['img2'] == cv2.warpAffine(original_img2, M, (w, h)))
    assert np.all(
        results['flow_fw_gt'] == cv2.warpAffine(original_flow_fw, M, (w, h)))
    assert np.all(
        results['flow_bw_gt'] == cv2.warpAffine(original_flow_bw, M, (w, h)))
    assert np.all(
        results['occ_fw_gt'] == cv2.warpAffine(original_occ_fw, M, (w, h)))
    assert np.all(
        results['occ_bw_gt'] == cv2.warpAffine(original_occ_bw, M, (w, h)))
    assert results['translate']


@pytest.mark.parametrize('gamma_range', [(0.7, 1.5), (1.2, 1.5), (0.5, 1.0)])
def test_random_gamma(gamma_range):
    # test gamma_range type
    with pytest.raises(AssertionError):
        transform = dict(type='RandomGamma', gamma_range=0.0)
        build_from_cfg(transform, PIPELINES)
    # test gamma_range < 0
    with pytest.raises(AssertionError):
        transform = dict(type='RandomGamma', gamma_range=[-1, 1])
        build_from_cfg(transform, PIPELINES)
    # test gamma[0] > gamma[1]
    with pytest.raises(AssertionError):
        transform = dict(type='RandomGamma', gamma_range=[2, 1])
        build_from_cfg(transform, PIPELINES)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()

    img_norm_cfg = dict(mean=[0., 0., 0.], std=[255., 255., 255.])

    # gamma correction on float32 image
    with pytest.raises(AssertionError):
        transforms = [
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomGamma', gamma_range=gamma_range)
        ]
        transforms = Compose(transforms)
        results = transforms(results)

    results, original_img1, original_img2, original_flow_fw, \
        original_flow_bw, original_occ_fw, original_occ_bw = \
        make_testdata_for_train()

    transform = [dict(type='RandomGamma', gamma_range=gamma_range)]
    transform = Compose(transform)
    results = transform(results)
    assert np.all(results['img1'].shape == original_img1.shape)
    assert np.all(results['img2'].shape == original_img2.shape)
    assert np.all(results['flow_fw_gt'] == original_flow_fw)
    assert np.all(results['flow_bw_gt'] == original_flow_bw)
    assert np.all(results['occ_fw_gt'] == original_occ_fw)
    assert np.all(results['occ_bw_gt'] == original_occ_bw)
    assert gamma_range[0] <= results['gamma'] <= gamma_range[1]

    results, original_img1, original_img2 = make_testdata_for_test()
    results = transform(results)
    assert np.all(results['img1'].shape == original_img1.shape)
    assert np.all(results['img2'].shape == original_img2.shape)
