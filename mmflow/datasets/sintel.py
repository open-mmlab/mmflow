# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset as MME_BaseDataset

from mmflow.registry import DATASETS
from .utils import get_data_filename, load_ann_info, load_img_info


@DATASETS.register_module()
class Sintel(MME_BaseDataset):
    """Sintel optical flow dataset.

    Args:
        pass_style (str): Pass style for Sintel dataset, and it has 2 options
            ['clean', 'final']. Default: 'clean'.
        scene (str, list, optional): Scene in Sintel dataset, if scene is None,
            it means collecting data in all of scene of Sintel dataset.
            Default: None.
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
            dataset. Defaults to None which means using all ``data_list``.
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
                 pass_style: str = 'clean',
                 scene: Optional[Union[str, Sequence[str]]] = None,
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

        assert pass_style in (
            'clean',
            'final'), '`pass_style` expected to be \'clean\' or \'final\' '
        f'but got {pass_style}'
        self.pass_style = pass_style

        self.scene = scene

        super().__init__(ann_file, metainfo, data_root, data_prefix,
                         filter_cfg, indices, serialize_data, pipeline,
                         test_mode, lazy_init, max_refetch)

    def load_data_list(self) -> List[dict]:
        """Load ``data_list``

        ``data_list`` can be load from an annotation file named as
        ``self.ann_file`` or by parsing dataset path.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        if self.ann_file.endswith('json'):
            # load data_list with annotation file
            return super().load_data_list()
        else:
            # load data_list by path parsing
            self.load_data_info()
            return self.data_list

    def filter_data(self) -> List[dict]:
        """Filter data_list according to ``scene`` and ``pass_style``

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
        return new_data_list

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""

        self._get_data_dir()

        img1_filenames = []
        img2_filenames = []
        flow_filenames = []
        occ_filenames = []
        invalid_filenames = []

        def get_filenames(data_dir, data_suffix, img_idx=None):
            data_filenames = []
            for data_dir_ in data_dir:
                data_filenames_ = get_data_filename(data_dir_, data_suffix)
                data_filenames_.sort()
                if img_idx == 1:
                    data_filenames += data_filenames_[:-1]
                elif img_idx == 2:
                    data_filenames += data_filenames_[1:]
                else:
                    data_filenames += data_filenames_
            return data_filenames

        img1_filenames = get_filenames(self.img1_dir, self.img1_suffix, 1)
        img2_filenames = get_filenames(self.img2_dir, self.img2_suffix, 2)
        load_img_info(self.data_list, img1_filenames, img2_filenames)

        if not self.test_mode:
            flow_filenames = get_filenames(self.flow_dir, self.flow_suffix)
            occ_filenames = get_filenames(self.occ_dir, self.occ_suffix)
            invalid_filenames = get_filenames(self.invalid_dir,
                                              self.invalid_suffix, 1)
            load_ann_info(self.data_list, flow_filenames, 'flow_fw_path')
            load_ann_info(self.data_list, occ_filenames, 'occ_fw_path')
            load_ann_info(self.data_list, invalid_filenames, 'invalid_path')

        for i in range(len(self.data_list)):
            self.data_list[i]['pass_style'] = self.pass_style
            self.data_list[i]['scene'] = self.data_list[i]['img1_path'].split(
                os.sep)[-2]
            if self.test_mode:
                self.data_list[i]['flow_fw_path'] = None
                self.data_list[i]['invalid_path'] = None
                self.data_list[i]['occ_fw_path'] = None

    def _get_data_dir(self) -> None:
        """Get the paths for images and optical flow."""
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'
        self.flow_suffix = '.flo'
        self.occ_suffix = '.png'
        self.invalid_suffix = '.png'

        self.subset_dir = osp.join('testing',
                                   'test') if self.test_mode else 'training'

        self.data_root = osp.join(self.data_root, self.subset_dir)

        img_root = osp.join(self.data_root, self.pass_style)
        all_scene = os.listdir(img_root)
        scene = all_scene if self.scene is None else self.scene
        scene = scene if isinstance(scene, (list, tuple)) else [scene]
        assert set(scene).issubset(set(all_scene))
        self.img1_dir = [osp.join(img_root, s) for s in scene]
        self.img2_dir = [osp.join(img_root, s) for s in scene]
        if not self.test_mode:
            flow_root = osp.join(self.data_root, 'flow')
            occ_root = osp.join(self.data_root, 'occlusions')
            invalid_root = osp.join(self.data_root, 'invalid')

            self.flow_dir = [osp.join(flow_root, s) for s in scene]
            self.occ_dir = [osp.join(occ_root, s) for s in scene]
            self.invalid_dir = [osp.join(invalid_root, s) for s in scene]
