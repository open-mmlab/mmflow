# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest
import torch
from mmcv.utils import Config, is_list_of
from mmengine.data import PixelData
from torch import Tensor

from mmflow.data import FlowDataSample
from mmflow.models import MaskFlowNetS, build_flow_estimator
from mmflow.models.flow_estimators.base import FlowEstimator
from mmflow.utils import register_all_modules

register_all_modules()


def _demo_model_inputs(H=64, W=64):
    batch_inputs = torch.randn(2, 6, H, W)
    batch_data_samples = []
    data_sample = FlowDataSample(metainfo={
        'img_shape': (H, W),
        'ori_shape': (H, W)
    })
    data_keys = ('gt_flow_fw', 'gt_flow_bw', 'gt_occ_fw', 'gt_occ_bw',
                 'gt_valid_fw')
    for key in data_keys:
        ch = 2 if 'flow' in key else 1
        data = PixelData(**dict(data=torch.randn(ch, H, W)))
        data_sample.set_data({key: data})
    batch_data_samples = [data_sample, data_sample]
    return batch_inputs, batch_data_samples


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

    batch_inputs, batch_data_samples = _demo_model_inputs()
    batch_inputs = batch_inputs.cuda()
    batch_data_samples = [
        data_samples.cuda() for data_samples in batch_data_samples
    ]
    # test tensor out
    if type(estimator) is MaskFlowNetS:
        out, _ = estimator(batch_inputs, batch_data_samples, mode='tensor')
    else:
        out = estimator(batch_inputs, batch_data_samples, mode='tensor')
    assert isinstance(out, dict)
    # test predict out
    out = estimator(batch_inputs, batch_data_samples, mode='predict')
    assert is_list_of(out, FlowDataSample)
    # test loss out
    loss = estimator(batch_inputs, batch_data_samples, mode='loss')
    assert float(loss['loss_flow']) > 0.


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

    batch_inputs, batch_data_samples = _demo_model_inputs()

    # test tensor out
    out = estimator(batch_inputs, batch_data_samples, mode='tensor')
    if cfg.model.type == 'RAFT':
        assert is_list_of(out, Tensor)
    else:
        assert isinstance(out, dict)

    # test predict out
    predict = estimator(batch_inputs, batch_data_samples, mode='predict')
    assert is_list_of(predict, FlowDataSample)

    # test loss out
    loss = estimator(batch_inputs, batch_data_samples, mode='loss')
    assert float(loss['loss_flow']) > 0.
