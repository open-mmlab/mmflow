# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from glob import glob
from typing import Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset as MME_BaseDataset

from mmflow.registry import DATASETS
from .utils import get_data_filename, load_ann_info, load_img_info


@DATASETS.register_module()
class FlyingThings3D(MME_BaseDataset):
    """FlyingThings3D subset dataset.

    Args:

        scene (str, optional): Scene in Flyingthings3D dataset, which has 3
            options None, 'left' and 'right', and None means ``data_list``
            will not be filter by scene filed. Defaults to 'left'.
            This default value is for RAFT, as FlyingThings3D is so large
            and not often used, and only RAFT use the 'left' data in it.
        pass_style (str): Pass style for FlyingThing3D dataset, and it has 2
            options 'clean', 'final'. Default: 'clean'.
        double (bool): Whether double dataset by change the images pairs from
            (img1, img2) to (img2, img1). Defaults to False.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img=None, ann=None).
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
    """
    METAINFO = dict()

    def __init__(self,
                 scene: Optional[str] = None,
                 pass_style: str = 'clean',
                 double: bool = False,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000) -> None:

        if scene is not None:
            assert scene in (
                'left', 'right'
            ), f'`scene` expected to be \'left\' or \'right\' but got {scene}'
        self.scene = scene

        assert pass_style in (
            'clean',
            'final'), '`pass_style` expected to be \'clean\' or \'final\' '
        f'but got {pass_style}'
        self.pass_style = pass_style
        self.double = double

        super().__init__(ann_file, metainfo, data_root, data_prefix,
                         filter_cfg, indices, serialize_data, pipeline,
                         test_mode, lazy_init, max_refetch)

    def load_data_list(self) -> List[dict]:
        """Load ``data_list``

        ``data_list`` can be load from an annotation file named as
        ``self.ann_file`` or by parsing dataset path.

        Returns:
            list[dict]: A list of annotation.
        """
        if self.ann_file.endswith('json'):
            # load data_list with annotation file
            return super().load_data_list()
        else:
            # load data_list by path parsing
            self.load_data_info()
            return self.data_list

    def filter_data(self) -> List[dict]:
        """Filter ``data_list`` according to ``scene``, ``pass_style`` and
        ``double``

        Returns:
            list[int]: Filtered results.
        """
        if self.scene is None:
            new_data_list = [
                data_info for data_info in self.data_list
                if data_info['pass_style'] == self.pass_style
            ]

        else:

            new_data_list = [
                data_info for data_info in self.data_list
                if data_info['scene'] == self.scene
                and data_info['pass_style'] == self.pass_style
            ]
        if self.double:
            changed_new_data_list = []
            for data_info in new_data_list:
                changed_data_info = copy.deepcopy(data_info)
                changed_data_info['img1_path'], changed_data_info[
                    'img2_path'] = data_info['img2_path'], data_info[
                        'img1_path']
                changed_data_info['flow_fw_path'], changed_data_info[
                    'flow_bw_path'] = data_info['flow_bw_path'], data_info[
                        'flow_fw_path']
                changed_new_data_list.append(changed_data_info)
            return new_data_list + changed_new_data_list

        else:
            return new_data_list

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""
        self._get_data_dir()

        img1_filenames = []
        img2_filenames = []
        flow_fw_filenames = []
        flow_bw_filenames = []

        for idir, fw_dir, bw_dir in zip(self.img1_dir, self.flow_fw_dir,
                                        self.flow_bw_dir):

            img_filenames_ = get_data_filename(idir, self.img1_suffix)
            img_filenames_.sort()
            flow_fw_filenames_ = get_data_filename(fw_dir, self.flow_suffix)
            flow_fw_filenames_.sort()
            flow_bw_filenames_ = get_data_filename(bw_dir, self.flow_suffix)
            flow_bw_filenames_.sort()

            img1_filenames += img_filenames_[:-1]
            img2_filenames += img_filenames_[1:]
            flow_fw_filenames += flow_fw_filenames_[:-1]
            flow_bw_filenames += flow_bw_filenames_[1:]

        load_img_info(self.data_list, img1_filenames, img2_filenames)

        load_ann_info(self.data_list, flow_fw_filenames, 'flow_fw_path')
        load_ann_info(self.data_list, flow_bw_filenames, 'flow_bw_path')

        for i in range(len(self.data_list)):
            self.data_list[i]['pass_style'] = self.pass_style
            self.data_list[i]['scene'] = 'left' if 'left' in self.data_list[i][
                'img1_path'] else 'right'

    def _get_data_dir(self) -> None:
        """Get the paths for images and optical flow."""

        self.flow_fw_dir = 'into_future'
        self.flow_bw_dir = 'into_past'
        self.flow_suffix = '.pfm'
        self.occ_suffix = '.png'
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'

        self.subset_dir = 'TEST' if self.test_mode else 'TRAIN'

        imgs_dirs = []
        flow_fw_dirs = []
        flow_bw_dirs = []

        if self.scene is None:
            scene = ['left', 'right']
        else:
            scene = [self.scene]

        pass_dir = 'frames_' + self.pass_style + 'pass'
        for i_scene in scene:
            imgs_dirs_ = glob(
                osp.join(self.data_root, pass_dir, self.subset_dir + '/*/*'))
            imgs_dirs_ = [osp.join(f, i_scene) for f in imgs_dirs_]
            imgs_dirs += imgs_dirs_

            flow_fw_dirs_ = glob(
                osp.join(self.data_root, 'optical_flow',
                         self.subset_dir + '/*/*/' + self.flow_fw_dir))

            flow_fw_dirs_ = [osp.join(f, i_scene) for f in flow_fw_dirs_]
            flow_fw_dirs += flow_fw_dirs_

            flow_bw_dirs_ = glob(
                osp.join(self.data_root, 'optical_flow',
                         self.subset_dir + '/*/*/' + self.flow_bw_dir))
            flow_bw_dirs_ = [osp.join(f, i_scene) for f in flow_bw_dirs_]
            flow_bw_dirs += flow_bw_dirs_

        self.img1_dir = imgs_dirs
        self.img2_dir = imgs_dirs
        self.flow_fw_dir = flow_fw_dirs
        self.flow_bw_dir = flow_bw_dirs
