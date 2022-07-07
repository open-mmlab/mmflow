# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pytest

from mmflow.datasets import read_flow
from mmflow.datasets.pipelines import LoadAnnotations
from mmflow.datasets.utils import read_flow_kitti
from mmflow.registry import TRANSFORMS

img1_ = osp.join(osp.dirname(__file__), '../data/00001_img1.ppm')
img2_ = osp.join(osp.dirname(__file__), '../data/00001_img2.ppm')
flow_ = osp.join(osp.dirname(__file__), '../data/00001_flow.flo')
flow_fw = osp.join(osp.dirname(__file__), '../data/0000000-flow_01.flo')
flow_bw = osp.join(osp.dirname(__file__), '../data/0000000-flow_10.flo')
occ_fw = osp.join(osp.dirname(__file__), '../data/0000000-occ_01.png')
occ_bw = osp.join(osp.dirname(__file__), '../data/0000000-occ_10.png')
sparse_flow_fw = osp.join(osp.dirname(__file__), '../data/sparse_flow.png')


class TestLoading:

    @pytest.mark.parametrize('to_float32', (False, True))
    def test_load_img(self, to_float32):
        transform_img = dict(type='LoadImageFromFile', to_float32=to_float32)
        results = dict(img1_path=img1_, img2_path=img2_)
        transform = TRANSFORMS.build(transform_img)
        results = transform(results)
        assert results['img1'].shape == (50, 50, 3)
        assert results['img2'].shape == (50, 50, 3)
        assert results['img_shape'] == (50, 50)
        assert results['ori_shape'] == (50, 50)
        if to_float32:
            assert results['img1'].dtype == np.float32
            assert results['img2'].dtype == np.float32
        else:
            assert np.all(results['img1'] == mmcv.imread(img1_))
            assert np.all(results['img2'] == mmcv.imread(img2_))
            assert results['img1'].dtype == np.uint8
            assert results['img2'].dtype == np.uint8

    def test_load_ann(self):
        # test single direction
        results = dict(flow_fw_path=flow_)
        transform = LoadAnnotations()
        results = transform(results)
        assert np.all(results['gt_flow_fw'] == read_flow(flow_))
        assert results['gt_flow_fw'].shape == (50, 50, 2)
        assert results['gt_flow_fw'].dtype == np.float32

        # test bidirection + occ
        results = dict(
            flow_fw_path=flow_fw,
            flow_bw_path=flow_bw,
            occ_fw_path=occ_fw,
            occ_bw_path=occ_bw)
        transform = LoadAnnotations(with_occ=True)
        results = transform(results)
        assert np.all(results['gt_flow_fw'] == read_flow(flow_fw))
        assert np.all(results['gt_flow_bw'] == read_flow(flow_bw))
        assert results['gt_flow_fw'].shape == (50, 50, 2)
        assert results['gt_flow_fw'].dtype == np.float32
        assert results['gt_flow_bw'].shape == (50, 50, 2)
        assert results['gt_flow_bw'].dtype == np.float32

        assert np.all(
            results['gt_occ_fw'] == (mmcv.imread(occ_fw, flag='grayscale') /
                                     255).astype(np.float32))
        assert np.all(
            results['gt_occ_bw'] == (mmcv.imread(occ_bw, flag='grayscale') /
                                     255).astype(np.float32))
        assert results['gt_occ_fw'].shape == (50, 50)
        assert results['gt_occ_fw'].dtype == np.float32
        assert results['gt_occ_bw'].shape == (50, 50)
        assert results['gt_occ_bw'].dtype == np.float32

        # test sparse
        results = dict(flow_fw_path=sparse_flow_fw)
        transform = LoadAnnotations(sparse=True)
        results = transform(results)
        flow, valid = read_flow_kitti(sparse_flow_fw)
        assert np.all(results['gt_valid_fw'] == valid)
        assert results['gt_valid_fw'].dtype == np.float32
        assert results['gt_valid_fw'].shape == (50, 50)
        assert results['gt_valid_bw'] is None
        assert np.all(results['gt_flow_fw'] == flow)
        assert results['gt_flow_fw'].shape == (50, 50, 2)
        assert results['gt_flow_fw'].dtype == np.float32
