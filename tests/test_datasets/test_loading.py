# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np

from mmflow.datasets.pipelines import LoadAnnotations, LoadImageFromFile

img1_ = osp.join(osp.dirname(__file__), '../data/00001_img1.ppm')
img2_ = osp.join(osp.dirname(__file__), '../data/00001_img2.ppm')
flow_ = osp.join(osp.dirname(__file__), '../data/00001_flow.flo')
flow_fw = osp.join(osp.dirname(__file__), '../data/0000000-flow_01.flo')
flow_bw = osp.join(osp.dirname(__file__), '../data/0000000-flow_10.flo')
occ_fw = osp.join(osp.dirname(__file__), '../data/0000000-occ_01.png')
occ_bw = osp.join(osp.dirname(__file__), '../data/0000000-occ_10.png')


class TestLoading:

    def test_load_img(self):
        results = dict(img_info=dict(filename1=img1_, filename2=img2_))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['filename1'] == img1_
        assert results['filename2'] == img2_
        assert results['ori_filename1'] == '00001_img1.ppm'
        assert results['ori_filename2'] == '00001_img2.ppm'
        assert results['img1'].shape == (50, 50, 3)
        assert results['img1'].dtype == np.uint8
        assert results['img2'].shape == (50, 50, 3)
        assert results['img2'].dtype == np.uint8
        assert results['img_shape'] == (50, 50, 3)
        assert results['ori_shape'] == (50, 50, 3)
        assert results['pad_shape'] == (50, 50, 3)
        assert np.all(results['scale_factor'] == np.array([1., 1.]))
        np.testing.assert_equal(results['img_norm_cfg']['mean'],
                                np.zeros(3, dtype=np.float32))
        assert repr(transform) == transform.__class__.__name__ + \
            "(to_float32=False,color_type='color',imdecode_backend='cv2')"

        # to_float32
        transform = LoadImageFromFile(to_float32=True)
        results = transform(copy.deepcopy(results))
        assert results['img1'].dtype == np.float32
        assert results['img2'].dtype == np.float32

    def test_load_ann(self):
        # test single direction
        results = dict(ann_info=dict(filename_flow=flow_), ann_fields=[])
        transform = LoadAnnotations()
        results = transform(copy.deepcopy(results))
        assert results['ann_fields'] == ['flow_gt']
        assert results['filename_flow'] == flow_
        assert results['ori_filename_flow'] == '00001_flow.flo'
        assert results['flow_gt'].shape == (50, 50, 2)
        assert results['flow_gt'].dtype == np.float32

        # test bidirection + occ

        results = dict(
            ann_info=dict(
                filename_flow_fw=flow_fw,
                filename_flow_bw=flow_bw,
                filename_occ_fw=occ_fw,
                filename_occ_bw=occ_bw),
            ann_fields=[])
        transform = LoadAnnotations(with_occ=True)
        results = transform(copy.deepcopy(results))
        assert results['ann_fields'] == [
            'flow_fw_gt', 'flow_bw_gt', 'occ_fw_gt', 'occ_bw_gt'
        ]
        assert results['filename_flow_fw'] == flow_fw
        assert results['filename_flow_bw'] == flow_bw
        assert results['ori_filename_flow_fw'] == '0000000-flow_01.flo'
        assert results['ori_filename_flow_bw'] == '0000000-flow_10.flo'
        assert results['filename_occ_fw'] == occ_fw
        assert results['filename_occ_bw'] == occ_bw
        assert results['ori_filename_occ_fw'] == '0000000-occ_01.png'
        assert results['ori_filename_occ_bw'] == '0000000-occ_10.png'

        assert results['flow_fw_gt'].shape == (50, 50, 2)
        assert results['flow_fw_gt'].dtype == np.float32
        assert results['flow_bw_gt'].shape == (50, 50, 2)
        assert results['flow_bw_gt'].dtype == np.float32

        assert results['occ_fw_gt'].shape == (50, 50)
        assert results['occ_fw_gt'].dtype == np.float32
        assert results['occ_bw_gt'].shape == (50, 50)
        assert results['occ_bw_gt'].dtype == np.float32
