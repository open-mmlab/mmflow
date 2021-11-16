# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmflow.datasets import build_dataset


def test_dataset_wrapper():
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

    concat_dataset_cfg = dict(
        type='ConcatDataset',
        datasets=[dataset_A_cfg, dataset_B_cfg],
        separate_eval=True)

    concat_dataset = build_dataset(concat_dataset_cfg)

    assert len(concat_dataset) == 4 * 2

    # TODO test separate_eval arguments

    dataset_A_repeat = dict(
        type='RepeatDataset', times=10, dataset=dataset_A_cfg)
    dataset_B_repeat = dict(
        type='RepeatDataset', times=5, dataset=dataset_B_cfg)

    repeat_datset = build_dataset([dataset_A_repeat, dataset_B_repeat])
    assert len(repeat_datset) == 4 * 15
