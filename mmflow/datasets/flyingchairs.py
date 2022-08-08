# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.dataset import BaseDataset as MME_BaseDataset

from mmflow.registry import DATASETS
from .utils import get_data_filename


@DATASETS.register_module()
class FlyingChairs(MME_BaseDataset):
    """FlyingChairs dataset.

    Args:
        split_file (str): File name of train-validation split file for
            FlyingChairs.
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
                 split_file: Optional[str] = None,
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
        self.split = split_file
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
            assert self.split is not None, 'At least one of `split_file` and '
            '`ann_file` must be input when FlyingChairs dataset initialized'
            self.split = np.loadtxt(self.split, dtype=np.int32).tolist()
            self.load_data_info()
            return self.data_list

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""

        # unpack FlyingChairs directly, will see `data` subdirctory.
        self.img1_dir = osp.join(self.data_root, 'data')
        self.img2_dir = osp.join(self.data_root, 'data')
        self.flow_dir = osp.join(self.data_root, 'data')

        # data in FlyingChairs dataset has specific suffix
        self.img1_suffix = '_img1.ppm'
        self.img2_suffix = '_img2.ppm'
        self.flow_suffix = '_flow.flo'

        img1_filenames = get_data_filename(self.img1_dir, self.img1_suffix)
        img2_filenames = get_data_filename(self.img2_dir, self.img2_suffix)
        flow_filenames = get_data_filename(self.flow_dir, self.flow_suffix)

        assert len(img1_filenames) == len(img2_filenames) == len(
            flow_filenames)

        self.load_img_info(img1_filenames, img2_filenames)
        self.load_ann_info(flow_filenames, 'filename_flow_fw')

    def load_img_info(self, img1_filename: Sequence[str],
                      img2_filename: Sequence[str]) -> None:
        """Load information of image1 and image2.

        Args:
            img1_filename (list): ordered list of abstract file path of img1.
            img2_filename (list): ordered list of abstract file path of img2.
        """

        num_file = len(img1_filename)
        for i in range(num_file):
            if (not self.test_mode
                    and self.split[i] == 1) or (self.test_mode
                                                and self.split[i] == 2):
                data_info = dict(
                    img1_path=img1_filename[i],
                    img2_path=img2_filename[i],
                )
                self.data_list.append(data_info)

    def load_ann_info(self, filename: Sequence[str],
                      filename_key: str) -> None:
        """Load information of optical flow.

        This function splits the dataset into two subsets, training subset and
        testing subset.

        Args:
            filename (list): ordered list of abstract file path of annotation.
            filename_key (str): the annotation e.g. 'flow'.
        """
        num_files = len(filename)
        num_tests = 0
        for i in range(num_files):
            if (not self.test_mode and self.split[i] == 1) \
                    or (self.test_mode and self.split[i] == 2):
                self.data_list[i - num_tests]['flow_fw_path'] = filename[i]
            else:
                num_tests += 1
