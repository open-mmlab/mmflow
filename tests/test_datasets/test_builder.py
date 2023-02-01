# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmflow.registry import DATASETS
from mmengine.registry import init_default_scope

init_default_scope('mmflow')
data_root = osp.join(osp.dirname(__file__), '../data/pseudo_sintel')

dataset_A_cfg = dict(
    type='Sintel',
    data_root=data_root,
    pipeline=[],
    test_mode=False,
    pass_style='clean')

dataset_B_cfg = dict(
    type='Sintel',
    data_root=data_root,
    pipeline=[],
    test_mode=False,
    pass_style='final')


@DATASETS.register_module()
class ToyDataset:

    def __init__(self, cnt=0):
        self.cnt = cnt

    def __item__(self, idx):
        return idx

    def __len__(self):
        return 100


def test_build_dataset():
    cfg = dict(type='ToyDataset')
    dataset = DATASETS.build(cfg)
    assert isinstance(dataset, ToyDataset)
    assert dataset.cnt == 0
    dataset = DATASETS.build(cfg, default_args=dict(cnt=1))
    assert isinstance(dataset, ToyDataset)
    assert dataset.cnt == 1
