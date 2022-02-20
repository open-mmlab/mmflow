# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re

import pytest

from mmflow.datasets import build_dataset


def test_flyingchairsocc():
    data_root = osp.join(
        osp.dirname(__file__), '../data/pseudo_flyingchairsocc')
    train_cfg = dict(type='FlyingChairsOcc', pipeline=[], data_root=data_root)
    train_dataset = build_dataset(train_cfg)
    assert len(train_dataset) == 1

    keys = list(train_dataset[0].keys())
    keys.sort()

    assert keys == [
        'ann_fields', 'ann_info', 'img1_dir', 'img2_dir', 'img_fields',
        'img_info'
    ]

    assert osp.split(train_dataset[0]['img_info']
                     ['filename1'])[-1] == f'{(0+1):05}_img1.png'
    assert osp.split(train_dataset[0]['img_info']
                     ['filename2'])[-1] == f'{(0+1):05}_img2.png'
    assert osp.split(train_dataset[0]['ann_info']
                     ['filename_flow_fw'])[-1] == f'{(0+1):05}_flow.flo'
    assert osp.split(train_dataset[0]['ann_info']
                     ['filename_flow_bw'])[-1] == f'{(0+1):05}_flow_b.flo'
    assert osp.split(train_dataset[0]['ann_info']
                     ['filename_occ_fw'])[-1] == f'{(0+1):05}_occ1.png'
    assert osp.split(train_dataset[0]['ann_info']
                     ['filename_occ_bw'])[-1] == f'{(0+1):05}_occ2.png'


def test_flyingchairs():

    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_flyingchairs')
    split_file = osp.join(
        osp.dirname(__file__),
        '../data/pseudo_flyingchairs/FlyingChairs_train_val.txt')
    train_cfg = dict(
        type='FlyingChairs',
        pipeline=[],
        data_root=data_root,
        split_file=split_file)
    train_dataset = build_dataset(train_cfg)

    test_cfg = dict(
        type='FlyingChairs',
        pipeline=[],
        test_mode=True,
        data_root=data_root,
        split_file=split_file)
    test_dataset = build_dataset(test_cfg)

    assert len(train_dataset) == 4
    assert len(test_dataset) == 1

    split = train_dataset.split
    idx_ = range(5)
    for idx in idx_:
        if split[idx] == 1:
            # test train datasets
            others = [i for i, val in enumerate(split[:idx]) if val == 2]
            others_nums = len(others)
            keys = list(train_dataset[idx - others_nums].keys())
            keys.sort()
            assert keys == [
                'ann_fields', 'ann_info', 'img1_dir', 'img2_dir', 'img_fields',
                'img_info'
            ]

            assert osp.split(train_dataset[idx - others_nums]['img_info']
                             ['filename1'])[-1] == f'{(idx+1):05}_img1.ppm'
            assert osp.split(train_dataset[idx - others_nums]['img_info']
                             ['filename2'])[-1] == f'{(idx+1):05}_img2.ppm'
            assert osp.split(train_dataset[idx - others_nums]['ann_info']
                             ['filename_flow'])[-1] == f'{(idx+1):05}_flow.flo'
        else:
            # test test datasets
            others = [i for i, val in enumerate(split[:idx]) if val == 1]
            others_nums = len(others)
            keys = list(test_dataset[idx - others_nums].keys())
            keys.sort()
            assert keys == [
                'ann_fields', 'ann_info', 'img1_dir', 'img2_dir', 'img_fields',
                'img_info'
            ]
            assert osp.split(test_dataset[idx - others_nums]['img_info']
                             ['filename1'])[-1] == f'{(idx+1):05}_img1.ppm'
            assert osp.split(test_dataset[idx - others_nums]['img_info']
                             ['filename2'])[-1] == f'{(idx+1):05}_img2.ppm'
            assert osp.split(test_dataset[idx - others_nums]['ann_info']
                             ['filename_flow'])[-1] == f'{(idx+1):05}_flow.flo'


@pytest.mark.parametrize(
    ('direction', 'scene'),
    [['forward', 'left'], ['bidirection', 'right'], ['backward', 'left']])
def test_flyingthings3d(direction, scene):

    data_root = osp.join(
        osp.dirname(__file__), '../data/pseudo_flyingthings3d')
    suffix_dir = dict(scene=scene, pass_style='clean')

    # test  invalid direction
    with pytest.raises(AssertionError):
        train = dict(
            type='FlyingThings3D',
            data_root=data_root,
            pipeline=[],
            test_mode=False,
            direction='A',
            **suffix_dir)
        train_dataset = build_dataset(train)

    if direction in ('forward', 'backward', 'bidirection'):
        train_len = 1
        test_len = 1
    else:
        train_len = 2
        test_len = 2

    def check(dataset, indices):
        for idx in indices:
            s1, filename1 = dataset[idx]['img_info']['filename1'].split(
                os.sep)[-2:]
            s2, filename2 = dataset[idx]['img_info']['filename2'].split(
                os.sep)[-2:]

            if direction == 'forward':
                sf, filename_flow = dataset[idx]['ann_info'][
                    'filename_flow'].split(os.sep)[-2:]
                assert s1 == s2 == sf == scene
                assert int(filename1[:-4]) == int(filename2[:-4]) - 1 == int(
                    re.findall(r'\d+', filename_flow)[0])
            elif direction == 'backward':
                sf, filename_flow = dataset[idx]['ann_info'][
                    'filename_flow'].split(os.sep)[-2:]
                assert s1 == s2 == sf == scene
                assert int(filename1[:-4]) == int(filename2[:-4]) + 1 == int(
                    re.findall(r'\d+', filename_flow)[0])
            elif direction == 'bidirection':
                sf_fw, filename_flow_fw = dataset[idx]['ann_info'][
                    'filename_flow_fw'].split(os.sep)[-2:]
                sf_bw, filename_flow_bw = dataset[idx]['ann_info'][
                    'filename_flow_bw'].split(os.sep)[-2:]
                assert s1 == s2 == sf_fw == sf_bw == scene
                assert int(filename1[:-4]) == int(filename2[:-4]) - 1 == int(
                    re.findall(r'\d+', filename_flow_fw)[0])
                assert int(filename1[:-4]) + 1 == int(filename2[:-4]) == int(
                    re.findall(r'\d+', filename_flow_bw)[0])

            else:
                sf, filename_flow = dataset[idx]['ann_info'][
                    'filename_flow'].split(os.sep)[-2:]
                assert s1 == s2 == sf == scene
                int(filename1[:-4]) == int(filename2[:-4]) - 1 == int(
                    re.findall(r'\d+', filename_flow)[0]) or int(
                        filename1[:-4]) + 1 == int(filename2[:-4]) == int(
                            re.findall(r'\d+', filename_flow)[0])

    # test training dataset
    train = dict(
        type='FlyingThings3D',
        data_root=data_root,
        pipeline=[],
        test_mode=False,
        direction=direction,
        **suffix_dir)
    train_dataset = build_dataset(train)

    assert len(train_dataset) == train_len
    train_idx = range(train_len)
    check(train_dataset, train_idx)
    # test testing dataset
    test = dict(
        type='FlyingThings3D',
        data_root=data_root,
        pipeline=[],
        test_mode=False,
        direction=direction,
        **suffix_dir)
    test_dataset = build_dataset(test)
    test_idx = range(test_len)
    check(test_dataset, test_idx)


@pytest.mark.parametrize(('direction', 'scene'), [('forward', 'left'),
                                                  ('backward', 'left'),
                                                  ('forward', 'right'),
                                                  ('backward', 'right'),
                                                  ('forward', None),
                                                  ('backward', None)])
def test_flyingthings3d_subset(direction, scene):

    data_root = osp.join(
        osp.dirname(__file__), '../data/pseudo_flyingthings3d_subset')

    train = dict(
        type='FlyingThings3DSubset',
        data_root=data_root,
        pipeline=[],
        test_mode=False,
        direction=direction,
        scene=scene)

    train_dataset = build_dataset(train)

    length_dataset = len(train_dataset)
    if direction is not None:
        if scene is None:
            assert length_dataset == 8
        elif scene == 'left':
            assert length_dataset == 4
        elif scene == 'right':
            assert length_dataset == 4
    else:
        if scene is None:
            assert length_dataset == 16
        elif scene == 'left':
            assert length_dataset == 8
        elif scene == 'right':
            assert length_dataset == 8
        length_dataset = int(length_dataset / 2)

    for idx in range(length_dataset):

        filename1 = osp.split(train_dataset[idx]['img_info']['filename1'])[-1]
        filename2 = osp.split(train_dataset[idx]['img_info']['filename2'])[-1]
        assert filename1[-4:] == '.png'
        assert filename2[-4:] == '.png'
        if direction == 'forward':
            filename_flow = osp.split(
                train_dataset[idx]['ann_info']['filename_flow'])[-1]
            filename_occ = osp.split(
                train_dataset[idx]['ann_info']['filename_occ'])[-1]
            assert filename_flow[-4:] == '.flo'
            assert filename1[:-4] == filename_flow[:-4] == filename_occ[:-4]
            assert int(filename1[:-4]) + 1 == int(filename2[:-4])

        elif direction == 'backward':
            filename_flow = osp.split(
                train_dataset[idx]['ann_info']['filename_flow'])[-1]
            filename_occ = osp.split(
                train_dataset[idx]['ann_info']['filename_occ'])[-1]
            assert filename_flow[-4:] == '.flo'
            assert filename1[:-4] == filename_flow[:-4] == filename_occ[:-4]
            assert int(filename1[:-4]) - 1 == int(filename2[:-4])

        elif direction == 'bidirection':
            filename_flow_fw = osp.split(
                train_dataset[idx]['ann_info']['filename_flow_fw'])[-1]
            filename_flow_bw = osp.split(
                train_dataset[idx]['ann_info']['filename_flow_bw'])[-1]
            filename_occ_fw = osp.split(
                train_dataset[idx]['ann_info']['filename_occ_fw'])[-1]
            filename_occ_bw = osp.split(
                train_dataset[idx]['ann_info']['filename_occ_bw'])[-1]
            assert filename1[-4:] == '.png'
            assert filename2[-4:] == '.png'
            assert filename_flow_fw[-4:] == '.flo'
            assert int(filename1[:-4]) - 1 == int(filename2[:-4])
            assert (filename1[:-4] == filename_flow_fw[:-4] ==
                    filename_occ_fw[:-4])
            assert (filename2[:-4] == filename_flow_bw[:-4] ==
                    filename_occ_bw[:-4])
        else:
            assert int(filename1[:-4]) + 1 == int(filename2[:-4])
            if scene is None:
                filename1 = osp.split(
                    train_dataset[idx + 8]['img_info']['filename1'])[-1]
                filename2 = osp.split(
                    train_dataset[idx + 8]['img_info']['filename2'])[-1]
                filename_flow = osp.split(
                    train_dataset[idx + 8]['ann_info']['filename_flow'])[-1]
                filename_occ = osp.split(
                    train_dataset[idx + 8]['ann_info']['filename_occ'])[-1]
            elif scene == 'left':
                filename1 = osp.split(
                    train_dataset[idx + 4]['img_info']['filename1'])[-1]
                filename2 = osp.split(
                    train_dataset[idx + 4]['img_info']['filename2'])[-1]
                filename_flow = osp.split(
                    train_dataset[idx + 4]['ann_info']['filename_flow'])[-1]
                filename_occ = osp.split(
                    train_dataset[idx + 4]['ann_info']['filename_occ'])[-1]

            elif scene == 'right':
                filename1 = osp.split(
                    train_dataset[idx + 4]['img_info']['filename1'])[-1]
                filename2 = osp.split(
                    train_dataset[idx + 4]['img_info']['filename2'])[-1]
                filename_flow = osp.split(
                    train_dataset[idx + 4]['ann_info']['filename_flow'])[-1]
                filename_occ = osp.split(
                    train_dataset[idx + 4]['ann_info']['filename_occ'])[-1]
            assert int(filename1[:-4]) - 1 == int(filename2[:-4])
            assert filename1[:-4] == filename_flow[:-4]


@pytest.mark.parametrize(('pstyle', 'scene'), [
    ('clean', None),
    ('final', None),
])
def test_sintel(pstyle, scene):
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_sintel')
    train = dict(
        type='Sintel',
        data_root=data_root,
        pipeline=[],
        test_mode=False,
        pass_style=pstyle,
        scene=scene)

    train_dataset = build_dataset(train)

    for i in range(len(train_dataset)):
        filename1 = osp.split(train_dataset[i]['img_info']['filename1'])[-1]
        filename2 = osp.split(train_dataset[i]['img_info']['filename2'])[-1]
        filename_flow = osp.split(
            train_dataset[i]['ann_info']['filename_flow'])[-1]
        assert filename1[-4:] == '.png'
        assert filename2[-4:] == '.png'
        assert filename_flow[-4:] == '.flo'
        assert filename1[:-4] == filename_flow[:-4]
        assert int(filename1[6:-4]) + 1 == int(filename2[6:-4])
        if scene is None:
            len(train_dataset) == 4


@pytest.mark.parametrize('test_mode', [True, False])
def test_kitti2012(test_mode):
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_kitti')
    train = dict(
        type='KITTI2012',
        pipeline=[],
        test_mode=test_mode,
        data_root=data_root)
    train_dataset = build_dataset(train)

    assert len(train_dataset) == 1

    filename1 = osp.split(train_dataset[0]['img_info']['filename1'])[-1]
    filename2 = osp.split(train_dataset[0]['img_info']['filename2'])[-1]
    assert filename1[-6:] == '10.png'
    assert filename2[-6:] == '11.png'
    assert filename1[:-6] == filename2[:-6]
    if not test_mode:
        filename_flow = osp.split(
            train_dataset[0]['ann_info']['filename_flow'])[-1]
        assert filename_flow[-6:] == '10.png'
        assert filename1[:-6] == filename2[:-6] == filename_flow[:-6]


@pytest.mark.parametrize('test_mode', [True, False])
def test_kitti2015(test_mode):
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_kitti')
    train = dict(
        type='KITTI2015',
        pipeline=[],
        test_mode=test_mode,
        data_root=data_root)
    train_dataset = build_dataset(train)

    assert len(train_dataset) == 1

    filename1 = osp.split(train_dataset[0]['img_info']['filename1'])[-1]
    filename2 = osp.split(train_dataset[0]['img_info']['filename2'])[-1]
    assert filename1[-6:] == '10.png'
    assert filename2[-6:] == '11.png'
    assert filename1[:-6] == filename2[:-6]
    if not test_mode:
        filename_flow = osp.split(
            train_dataset[0]['ann_info']['filename_flow'])[-1]
        assert filename_flow[-6:] == '10.png'
        assert filename1[:-6] == filename2[:-6] == filename_flow[:-6]


@pytest.mark.parametrize('test_mode', [True, False])
def test_hd1k(test_mode):
    train = dict(
        type='HD1K',
        pipeline=[],
        test_mode=test_mode,
        data_root=osp.join(osp.dirname(__file__), '../data/pseudo_hd1k'))

    train_dataset = build_dataset(train)
    assert len(train_dataset) == 1

    filename1 = osp.split(train_dataset[0]['img_info']['filename1'])[-1]
    filename2 = osp.split(train_dataset[0]['img_info']['filename2'])[-1]
    assert filename1[:6] == filename2[:6]
    assert int(filename1[7:11]) + 1 == int(filename2[7:11])
    if not test_mode:
        filename_flow = osp.split(
            train_dataset[0]['ann_info']['filename_flow'])[-1]
        assert filename1 == filename_flow


@pytest.mark.parametrize('test_mode', [True, False])
def test_chairssdhom(test_mode):
    train = dict(
        type='ChairsSDHom',
        pipeline=[],
        test_mode=test_mode,
        data_root=osp.join(
            osp.dirname(__file__), '../data/pseudo_chairssdhom'))
    train_dataset = build_dataset(train)
    assert len(train_dataset) == 1
    filename1 = osp.split(train_dataset[0]['img_info']['filename1'])[-1]
    filename2 = osp.split(train_dataset[0]['img_info']['filename2'])[-1]
    assert filename1 == filename2
    filename_flow = osp.split(
        train_dataset[0]['ann_info']['filename_flow'])[-1]
    assert filename1[:-3] == filename_flow[:-3]
