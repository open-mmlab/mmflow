# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest

from mmflow.datasets import (HD1K, KITTI2012, KITTI2015, ChairsSDHom,
                             FlyingChairs, FlyingChairsOcc, FlyingThings3D,
                             FlyingThings3DSubset, Sintel)


class TestFlyingChiars:

    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_flyingchairs')

    @pytest.mark.parametrize('init_function', ('ann_file', 'path_parse'))
    def test_load_data_list(self, init_function):

        if init_function == 'ann_file':
            train_dataset, test_dataset = self._load_annotation_file()
        else:
            train_dataset, test_dataset = self._load_path_parsing()

        assert len(train_dataset) == 4
        assert len(test_dataset) == 1

        split = np.loadtxt(
            osp.join(self.data_root, 'FlyingChairs_train_val.txt'),
            dtype=np.int32).tolist()

        for idx, i_split in enumerate(split):
            if i_split == 1:
                # test train datasets
                others = [i for i, val in enumerate(split[:idx]) if val == 2]
                others_nums = len(others)
                data_info = train_dataset[idx - others_nums]

            else:
                # test test datasets
                others = [i for i, val in enumerate(split[:idx]) if val == 1]
                others_nums = len(others)
                data_info = test_dataset[idx - others_nums]
            keys = list(data_info.keys())
            assert set(['img1_path', 'img2_path',
                        'flow_fw_path']).issubset(set(keys))

            assert osp.split(
                data_info['img1_path'])[-1] == f'{(idx+1):05}_img1.ppm'
            assert osp.split(
                data_info['img2_path'])[-1] == f'{(idx+1):05}_img2.ppm'
            assert osp.split(
                data_info['flow_fw_path'])[-1] == f'{(idx+1):05}_flow.flo'

    def _load_annotation_file(self):
        train_ann_file = osp.join(self.data_root, 'train.json')
        test_ann_file = osp.join(self.data_root, 'test.json')
        train_cfg = dict(
            pipeline=[], data_root=self.data_root, ann_file=train_ann_file)
        train_dataset = FlyingChairs(**train_cfg)

        test_cfg = dict(
            pipeline=[],
            test_mode=True,
            data_root=self.data_root,
            ann_file=test_ann_file)
        test_dataset = FlyingChairs(**test_cfg)
        return train_dataset, test_dataset

    def _load_path_parsing(self):
        # Test split_file and ann_file are not defined.
        with pytest.raises(AssertionError):
            FlyingChairs(split_file=None, ann_file='')

        split_file = osp.join(self.data_root, 'FlyingChairs_train_val.txt')
        train_cfg = dict(
            pipeline=[], data_root=self.data_root, split_file=split_file)
        train_dataset = FlyingChairs(**train_cfg)

        test_cfg = dict(
            pipeline=[],
            test_mode=True,
            data_root=self.data_root,
            split_file=split_file)
        test_dataset = FlyingChairs(**test_cfg)
        return train_dataset, test_dataset


class TestFlyingThings3D:
    data_root = osp.join(
        osp.dirname(__file__), '../data/pseudo_flyingthings3d')

    @pytest.mark.parametrize('init_function', ('ann_file', 'path_parse'))
    @pytest.mark.parametrize('scene', ('left', 'right', None))
    @pytest.mark.parametrize('pass_style', ('clean', 'final'))
    @pytest.mark.parametrize('double', (True, False))
    def test_load_data_list(self, init_function, scene, pass_style, double):

        self.scene = scene
        self.pass_style = pass_style
        self.double = double
        if init_function == 'ann_file':
            train_dataset, test_dataset = self._load_annotation_file()
        else:
            train_dataset, test_dataset = self._load_path_parsing()

        multiple_param = 2 if double else 1
        for dataset in (train_dataset, test_dataset):
            if self.scene is None:
                assert len(dataset) == multiple_param * 2
            else:
                assert len(dataset) == multiple_param * 1
            for data_info in dataset:
                img1_path = data_info['img1_path']
                img2_path = data_info['img2_path']
                flow_fw_path = data_info['flow_fw_path']
                flow_bw_path = data_info['flow_bw_path']

                if self.scene is not None:
                    assert self.scene == data_info['scene']

                test_scene = data_info['scene']
                assert (test_scene in img1_path and test_scene in img2_path
                        and test_scene in flow_fw_path
                        and test_scene in flow_bw_path)
                assert self.pass_style == data_info['pass_style']
                assert (data_info['pass_style'] in img1_path
                        and data_info['pass_style'] in img2_path)
                assert img1_path[-8:-4] == flow_fw_path[-10:-6]
                assert img2_path[-8:-4] == flow_bw_path[-10:-6]

    def _load_annotation_file(self):
        train_ann_file = osp.join(self.data_root, 'train.json')
        test_ann_file = osp.join(self.data_root, 'test.json')
        train_cfg = dict(
            pipeline=[],
            data_root=self.data_root,
            ann_file=train_ann_file,
            pass_style=self.pass_style,
            scene=self.scene,
            double=self.double)
        test_cfg = dict(
            test_mode=True,
            pipeline=[],
            data_root=self.data_root,
            ann_file=test_ann_file,
            pass_style=self.pass_style,
            scene=self.scene,
            double=self.double)
        return FlyingThings3D(**train_cfg), FlyingThings3D(**test_cfg)

    def _load_path_parsing(self):
        train_cfg = dict(
            pipeline=[],
            data_root=self.data_root,
            pass_style=self.pass_style,
            scene=self.scene,
            double=self.double)
        test_cfg = dict(
            pipeline=[],
            test_mode=True,
            data_root=self.data_root,
            pass_style=self.pass_style,
            scene=self.scene,
            double=self.double)
        return FlyingThings3D(**train_cfg), FlyingThings3D(**test_cfg)


class TestFlyingThings3DSubset:
    data_root = osp.join(
        osp.dirname(__file__), '../data/pseudo_flyingthings3d_subset')

    def test_init(self):
        # test invalid scene
        with pytest.raises(AssertionError):
            FlyingThings3DSubset(scene='a')

    @pytest.mark.parametrize('init_function', ('ann_file', 'path_parse'))
    @pytest.mark.parametrize('scene', ('left', 'right', None))
    def test_load_data_list(self, init_function, scene):
        self.scene = scene
        if init_function == 'ann_file':
            train_dataset, test_dataset = self._load_annotation_file()
        else:
            train_dataset, test_dataset = self._load_path_parsing()

        for dataset in (train_dataset, test_dataset):
            if self.scene is None:
                assert len(dataset) == 8
            elif self.scene == 'left':
                assert len(dataset) == 4
            elif self.scene == 'right':
                assert len(dataset) == 4
                for data_info in dataset:
                    img1_path = data_info['img1_path'].split(osp.sep)[-1]
                    img2_path = data_info['img2_path'].split(osp.sep)[-1]
                    flow_fw_path = data_info['flow_fw_path'].split(osp.sep)[-1]
                    flow_bw_path = data_info['flow_bw_path'].split(osp.sep)[-1]
                    occ_fw_path = data_info['occ_fw_path'].split(osp.sep)[-1]
                    occ_bw_path = data_info['occ_bw_path'].split(osp.sep)[-1]
                    assert int(img1_path[:-4]) + 1 == int(img2_path[:-4])
                    assert (img1_path[:-4] == flow_fw_path[:-4] ==
                            occ_fw_path[:-4])
                    assert (img2_path[:-4] == flow_bw_path[:-4] ==
                            occ_bw_path[:-4])
                    assert (flow_fw_path[-4:] == '.flo'
                            and flow_bw_path[-4:] == '.flo')

    def _load_annotation_file(self):
        train_ann_file = osp.join(self.data_root, 'train.json')
        test_ann_file = osp.join(self.data_root, 'test.json')
        train_cfg = dict(
            ann_file=train_ann_file,
            data_root=self.data_root,
            pipeline=[],
            test_mode=False,
            scene=self.scene)
        test_cfg = dict(
            ann_file=test_ann_file,
            data_root=self.data_root,
            pipeline=[],
            test_mode=True,
            scene=self.scene)
        return FlyingThings3DSubset(**train_cfg), FlyingThings3DSubset(
            **test_cfg)

    def _load_path_parsing(self):
        train_cfg = dict(
            data_root=self.data_root,
            pipeline=[],
            test_mode=False,
            scene=self.scene)
        test_cfg = dict(
            ann_file='',
            data_root=self.data_root,
            pipeline=[],
            test_mode=True,
            scene=self.scene)
        return FlyingThings3DSubset(**train_cfg), FlyingThings3DSubset(
            **test_cfg)


class TestSintel:
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_sintel')

    def test_init(self):
        # test invalid pass_style
        with pytest.raises(AssertionError):
            Sintel(pass_style='a')

    @pytest.mark.parametrize('init_function', ('ann_file', 'path_parse'))
    @pytest.mark.parametrize('pass_style', ('clean', 'final'))
    def test_load_data_list(self, init_function, pass_style):
        self.pass_style = pass_style
        if init_function == 'ann_file':
            train_dataset, test_dataset = self._load_annotation_file()
        else:
            train_dataset, test_dataset = self._load_path_parsing()

        for dataset in (train_dataset, test_dataset):
            assert len(dataset) == 2
            # assert dataset.metainfo['pass_style'] == self.pass_style
            for data_info in dataset:
                img1_path = data_info['img1_path']
                img2_path = data_info['img2_path']
                assert int(img1_path[-8:-4]) + 1 == int(img2_path[-8:-4])
                assert self.pass_style in img1_path
                if not dataset.test_mode:
                    flow_fw_path = data_info['flow_fw_path']
                    occ_fw_path = data_info['occ_fw_path']
                    invalid_path = data_info['invalid_path']
                    assert flow_fw_path[
                        -4:] == '.flo' and flow_fw_path[-8:-4] == img1_path[
                            -8:-4] == occ_fw_path[-8:-4] == invalid_path[-8:-4]

    def _load_annotation_file(self):
        train_cfg = dict(
            data_root=self.data_root,
            test_mode=False,
            ann_file='train.json',
            pass_style=self.pass_style,
        )
        test_cfg = dict(
            data_root=self.data_root,
            test_mode=True,
            ann_file='test.json',
            pass_style=self.pass_style,
        )
        return Sintel(**train_cfg), Sintel(**test_cfg)

    def _load_path_parsing(self):
        train_cfg = dict(
            data_root=self.data_root,
            test_mode=False,
            pass_style=self.pass_style,
        )
        test_cfg = dict(
            data_root=self.data_root,
            test_mode=True,
            pass_style=self.pass_style,
        )
        return Sintel(**train_cfg), Sintel(**test_cfg)


class TestKITTI2015:
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_kitti')

    @pytest.mark.parametrize('init_function', ('ann_file', 'path_parse'))
    def test_load_data_list(self, init_function):
        if init_function == 'ann_file':
            dataset = self._load_annotation_file()
        else:
            dataset = self._load_path_parsing()
        assert len(dataset) == 1

        for data_info in dataset:
            img1_path = data_info['img1_path']
            img2_path = data_info['img2_path']
            flow_fw_path = data_info['flow_fw_path']
            assert img1_path[-6:] == '10.png'
            assert img2_path[-6:] == '11.png'
            assert flow_fw_path[-6:] == '10.png'

    def _load_annotation_file(self):
        dataset_cfg = dict(
            data_root=self.data_root,
            test_mode=False,
            ann_file='2015_train.json')

        return KITTI2015(**dataset_cfg)

    def _load_path_parsing(self):
        dataset_cfg = dict(data_root=self.data_root, test_mode=False)

        return KITTI2015(**dataset_cfg)


class TestKITTI2012:
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_kitti')

    @pytest.mark.parametrize('init_function', ('ann_file', 'path_parse'))
    def test_load_data_list(self, init_function):
        if init_function == 'ann_file':
            dataset = self._load_annotation_file()
        else:
            dataset = self._load_path_parsing()
        assert len(dataset) == 1

        for data_info in dataset:
            img1_path = data_info['img1_path']
            img2_path = data_info['img2_path']
            flow_fw_path = data_info['flow_fw_path']
            assert img1_path[-6:] == '10.png'
            assert img2_path[-6:] == '11.png'
            assert flow_fw_path[-6:] == '10.png'

    def _load_annotation_file(self):
        dataset_cfg = dict(
            data_root=self.data_root,
            test_mode=False,
            ann_file='2012_train.json')

        return KITTI2012(**dataset_cfg)

    def _load_path_parsing(self):
        dataset_cfg = dict(data_root=self.data_root, test_mode=False)

        return KITTI2012(**dataset_cfg)


class TestFlyingChairsOcc:
    data_root = osp.join(
        osp.dirname(__file__), '../data/pseudo_flyingchairsocc')

    @pytest.mark.parametrize('init_function', ('ann_file', 'path_parse'))
    def test_load_data_list(self, init_function):

        if init_function == 'ann_file':
            train_dataset, test_dataset = self._load_annotation_file()
        else:
            train_dataset, test_dataset = self._load_path_parsing()

        assert len(train_dataset) == 1
        assert len(test_dataset) == 1

        split = np.loadtxt(
            osp.join(self.data_root, 'FlyingChairsOcc_train_val.txt'),
            dtype=np.int32).tolist()

        for idx, i_split in enumerate(split):
            if i_split == 1:
                # test train datasets
                others = [i for i, val in enumerate(split[:idx]) if val == 2]
                others_nums = len(others)
                data_info = train_dataset[idx - others_nums]

            else:
                # test test datasets
                others = [i for i, val in enumerate(split[:idx]) if val == 1]
                others_nums = len(others)
                data_info = test_dataset[idx - others_nums]
            keys = list(data_info.keys())
            assert set(['img1_path', 'img2_path',
                        'flow_fw_path']).issubset(set(keys))

            assert osp.split(
                data_info['img1_path'])[-1] == f'{(idx+1):05}_img1.png'
            assert osp.split(
                data_info['img2_path'])[-1] == f'{(idx+1):05}_img2.png'
            assert osp.split(
                data_info['flow_fw_path'])[-1] == f'{(idx+1):05}_flow.flo'
            assert osp.split(
                data_info['flow_bw_path'])[-1] == f'{(idx+1):05}_flow_b.flo'
            assert osp.split(
                data_info['occ_fw_path'])[-1] == f'{(idx+1):05}_occ1.png'
            assert osp.split(
                data_info['occ_bw_path'])[-1] == f'{(idx+1):05}_occ2.png'

    def _load_annotation_file(self):
        train_ann_file = osp.join(self.data_root, 'train.json')
        test_ann_file = osp.join(self.data_root, 'test.json')
        train_cfg = dict(
            pipeline=[], data_root=self.data_root, ann_file=train_ann_file)
        train_dataset = FlyingChairs(**train_cfg)

        test_cfg = dict(
            pipeline=[],
            test_mode=True,
            data_root=self.data_root,
            ann_file=test_ann_file)
        test_dataset = FlyingChairs(**test_cfg)
        return train_dataset, test_dataset

    def _load_path_parsing(self):

        split_file = osp.join(self.data_root, 'FlyingChairsOcc_train_val.txt')
        train_cfg = dict(
            pipeline=[], data_root=self.data_root, split_file=split_file)
        train_dataset = FlyingChairsOcc(**train_cfg)

        test_cfg = dict(
            pipeline=[],
            test_mode=True,
            data_root=self.data_root,
            split_file=split_file)
        test_dataset = FlyingChairsOcc(**test_cfg)
        return train_dataset, test_dataset


class TestHD1K:
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_hd1k')

    @pytest.mark.parametrize('init_function', ('ann_file', 'path_parse'))
    def test_load_data_list(self, init_function):

        if init_function == 'ann_file':
            dataset = self._load_annotation_file()
        else:
            dataset = self._load_path_parsing()
        assert len(dataset) == 1

        for data_info in dataset:
            img1_path = data_info['img1_path']
            img2_path = data_info['img2_path']
            flow_fw_path = data_info['flow_fw_path']
            assert img1_path[-8:] == '0010.png'
            assert img2_path[-8:] == '0011.png'
            assert flow_fw_path[-8:] == '0010.png'

    def _load_annotation_file(self):
        dataset_cfg = dict(
            data_root=self.data_root,
            test_mode=False,
            ann_file='hd1k_train.json')

        return HD1K(**dataset_cfg)

    def _load_path_parsing(self):
        dataset_cfg = dict(data_root=self.data_root, test_mode=False)

        return HD1K(**dataset_cfg)


class TestChairsSDHom:
    data_root = osp.join(osp.dirname(__file__), '../data/pseudo_chairssdhom')

    @pytest.mark.parametrize('init_function', ('ann_file', 'path_parse'))
    @pytest.mark.parametrize('metainfo', ('empty', 'valid'))
    def test_load_data_list(self, init_function, metainfo):
        if init_function == 'ann_file':
            train_dataset, test_dataset = self._load_annotation_file()
        else:
            if metainfo == 'empty':
                train_dataset, test_dataset = self._load_path_parsing()
            else:
                local_metainfo = dict(dataset='ChairsSDHom')
                train_dataset, test_dataset = self._load_path_parsing(
                    local_metainfo)

        assert len(train_dataset) == 1
        assert len(test_dataset) == 1

        # Test if the filenames of img1, img2 and flow are the same.
        for dataset in [train_dataset, test_dataset]:
            for data_info in dataset:
                img1_path = data_info['img1_path']
                img2_path = data_info['img2_path']
                flow_fw_path = data_info['flow_fw_path']
                assert int(osp.splitext(osp.basename(img1_path))[0]) \
                       == int(osp.splitext(osp.basename(img2_path))[0])\
                       == int(osp.splitext(osp.basename(flow_fw_path))[0])

    def _load_annotation_file(self):
        train_ann_file = osp.join(self.data_root, 'train.json')
        test_ann_file = osp.join(self.data_root, 'test.json')
        train_cfg = dict(data_root=self.data_root, ann_file=train_ann_file)
        train_dataset = ChairsSDHom(**train_cfg)

        test_cfg = dict(
            test_mode=True, data_root=self.data_root, ann_file=test_ann_file)
        test_dataset = ChairsSDHom(**test_cfg)
        return train_dataset, test_dataset

    def _load_path_parsing(self, metainfo=None):
        train_cfg = dict(data_root=self.data_root, metainfo=metainfo)
        train_dataset = ChairsSDHom(**train_cfg)

        test_cfg = dict(
            test_mode=True, data_root=self.data_root, metainfo=metainfo)
        test_dataset = ChairsSDHom(**test_cfg)
        return train_dataset, test_dataset
