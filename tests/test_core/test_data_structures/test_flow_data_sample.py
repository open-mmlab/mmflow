# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import PixelData

from mmflow.core import FlowDataSample


class TestFlowDataSample(TestCase):

    def test_init(self):
        meta_info = dict(img_size=[256, 256])
        data_sample = FlowDataSample(metainfo=meta_info)
        assert 'img_size' in data_sample
        assert data_sample.img_size == [256, 256]
        assert data_sample.get('img_size') == [256, 256]

    def test_setter(self):
        img_meta = dict(img_shape=(4, 5, 3))
        data_sample = FlowDataSample(metainfo=img_meta)
        # test ground truth
        flow_fw = torch.rand((2, 4, 5))
        gt_flow_fw = PixelData(metainfo=img_meta)
        gt_flow_fw.data = flow_fw

        flow_bw = torch.rand((2, 4, 5))
        gt_flow_bw = PixelData(metainfo=img_meta)
        gt_flow_bw.data = flow_bw

        occ_fw = torch.rand((1, 4, 5))
        gt_occ_fw = PixelData(metainfo=img_meta)
        gt_occ_fw.data = occ_fw

        occ_bw = torch.rand((1, 4, 5))
        gt_occ_bw = PixelData(metainfo=img_meta)
        gt_occ_bw.data = occ_bw

        valid_fw = torch.rand((1, 4, 5))
        gt_valid_fw = PixelData(metainfo=img_meta)
        gt_valid_fw.data = valid_fw
        valid_bw = torch.rand((1, 4, 5))
        gt_valid_bw = PixelData(metainfo=img_meta)
        gt_valid_bw.data = valid_bw

        data_sample.gt_flow_fw = gt_flow_fw
        data_sample.gt_flow_bw = gt_flow_bw
        data_sample.gt_occ_fw = gt_occ_fw
        data_sample.gt_occ_bw = gt_occ_bw
        data_sample.gt_valid_fw = gt_valid_fw
        data_sample.gt_valid_bw = gt_valid_bw

        torch.equal(data_sample.gt_flow_fw.data, flow_fw)
        torch.equal(data_sample.gt_flow_bw.data, flow_bw)
        torch.equal(data_sample.gt_occ_fw.data, occ_fw)
        torch.equal(data_sample.gt_occ_bw.data, occ_bw)
        torch.equal(data_sample.gt_valid_fw.data, valid_fw)
        torch.equal(data_sample.gt_valid_bw.data, valid_bw)

        # test prediction
        pred_flow_fw = PixelData(metainfo=img_meta)
        pred_flow_fw.data = flow_fw

        pred_flow_bw = PixelData(metainfo=img_meta)
        pred_flow_bw.data = flow_bw

        pred_occ_fw = PixelData(metainfo=img_meta)
        pred_occ_fw.data = occ_fw

        pred_occ_bw = PixelData(metainfo=img_meta)
        pred_occ_bw.data = occ_bw

        data_sample.pred_flow_fw = pred_flow_fw
        data_sample.pred_flow_bw = pred_flow_bw
        data_sample.pred_occ_fw = pred_occ_fw
        data_sample.pred_occ_bw = pred_occ_bw

        torch.equal(data_sample.pred_flow_fw.data, flow_fw)
        torch.equal(data_sample.pred_flow_bw.data, flow_bw)
        torch.equal(data_sample.pred_occ_fw.data, occ_fw)
        torch.equal(data_sample.pred_occ_bw.data, occ_bw)

    def test_deleter(self):
        img_meta = dict(img_shape=(4, 5, 3))
        data_sample = FlowDataSample(metainfo=img_meta)

        gt_flow_fw = PixelData(metainfo=img_meta)
        gt_flow_bw = PixelData(metainfo=img_meta)
        gt_occ_fw = PixelData(metainfo=img_meta)
        gt_occ_bw = PixelData(metainfo=img_meta)
        gt_valid_fw = PixelData(metainfo=img_meta)
        gt_valid_bw = PixelData(metainfo=img_meta)

        pred_flow_fw = PixelData(metainfo=img_meta)
        pred_flow_bw = PixelData(metainfo=img_meta)
        pred_occ_fw = PixelData(metainfo=img_meta)
        pred_occ_bw = PixelData(metainfo=img_meta)

        data_sample.gt_flow_fw = gt_flow_fw
        data_sample.gt_flow_bw = gt_flow_bw
        data_sample.gt_occ_fw = gt_occ_fw
        data_sample.gt_occ_bw = gt_occ_bw
        data_sample.gt_valid_fw = gt_valid_fw
        data_sample.gt_valid_bw = gt_valid_bw
        data_sample.pred_flow_fw = pred_flow_fw
        data_sample.pred_flow_bw = pred_flow_bw
        data_sample.pred_occ_fw = pred_occ_fw
        data_sample.pred_occ_bw = pred_occ_bw

        del data_sample.gt_flow_fw
        assert 'gt_flow_fw' not in data_sample
        assert not hasattr(data_sample, 'gt_flow_fw')

        del data_sample.gt_flow_bw
        assert 'gt_flow_bw' not in data_sample
        assert not hasattr(data_sample, 'gt_flow_bw')

        del data_sample.gt_occ_fw
        assert 'gt_occ_fw' not in data_sample
        assert not hasattr(data_sample, 'gt_occ_fw')

        del data_sample.gt_occ_bw
        assert 'gt_occ_bw' not in data_sample
        assert not hasattr(data_sample, 'gt_occ_bw')

        del data_sample.gt_valid_fw
        assert 'gt_valid_fw' not in data_sample
        assert not hasattr(data_sample, 'gt_valid_fw')

        del data_sample.gt_valid_bw
        assert 'gt_valid_bw' not in data_sample
        assert not hasattr(data_sample, 'gt_valid_bw')

        del data_sample.pred_flow_fw
        assert 'pred_flow_fw' not in data_sample
        assert not hasattr(data_sample, 'pred_flow_fw')

        del data_sample.pred_flow_bw
        assert 'pred_flow_bw' not in data_sample
        assert not hasattr(data_sample, 'pred_flow_bw')

        del data_sample.pred_occ_fw
        assert 'pred_occ_fw' not in data_sample
        assert not hasattr(data_sample, 'pred_occ_fw')

        del data_sample.pred_occ_bw
        assert 'pred_occ_bw' not in data_sample
        assert not hasattr(data_sample, 'pred_occ_bw')
