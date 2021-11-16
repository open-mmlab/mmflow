# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union

import mmcv
from torch.utils.data import Dataset

from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    Args:
        data_root (str): Directory for dataset.
        pipeline (Sequence[dict]): Processing pipeline.
        test_mode (bool): Whether the dataset works for model testing or
            training.
    """

    def __init__(self,
                 data_root: str,
                 pipeline: Sequence[dict],
                 test_mode: bool = False) -> None:
        super().__init__()
        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.dataset_name = self.__class__.__name__
        """
        data_infos is the list of data_info containing img_info and ann_info
        data_info
          - img_info
              - filename1
              - filename2
          - ann_info
              - filename_key
        key might be flow, flow_fw, flow_bw, occ, occ_fw, occ_bw, valid
        """
        self.data_infos = []

        self.load_data_info()

    @abstractmethod
    def load_data_info(self):
        """Placeholder for load data information."""
        pass

    def pre_pipeline(self, results: dict) -> None:
        """Prepare results dict for pipeline."""

        results['img_fields'] = ['img1', 'img2']
        results['ann_fields'] = []
        results['img1_dir'] = self.img1_dir
        results['img2_dir'] = self.img2_dir

    def prepare_data(self, idx: int) -> dict:
        """Get data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        results = copy.deepcopy(self.data_infos[idx])
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> dict:
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        return self.prepare_data(idx)

    @staticmethod
    def load_img_info(data_infos: Sequence[dict], img1_filename: Sequence[str],
                      img2_filename: Sequence[str]) -> None:
        """Load information of images.

        Args:
            data_infos (list): data information.
            img1_filename (list): ordered list of abstract file path of img1.
            img2_filename (list): ordered list of abstract file path of img2.
        """

        num_file = len(img1_filename)
        for i in range(num_file):
            data_info = dict(
                img_info=dict(
                    filename1=img1_filename[i], filename2=img2_filename[i]),
                ann_info=dict())
            data_infos.append(data_info)

    @staticmethod
    def load_ann_info(data_infos: Sequence[dict], filename: Sequence[str],
                      filename_key: str) -> None:
        """Load information of annotation.

        Args:
            data_infos (list): data information.
            filename (list): ordered list of abstract file path of annotation.
            filename_key (str): the annotation key e.g. 'flow'.
        """
        assert len(filename) == len(data_infos)
        num_files = len(filename)
        for i in range(num_files):
            data_infos[i]['ann_info'][filename_key] = filename[i]

    @staticmethod
    def get_data_filename(
            data_dirs: Union[Sequence[str], str],
            suffix: Optional[str] = None,
            exclude: Optional[Sequence[str]] = None) -> Sequence[str]:
        """Get file name from data directory.

        Args:
            data_dirs (list, str): the directory of data
            suffix (str, optional): the suffix for data file. Defaults to None.
            exclude (list, optional): list of files will be excluded.

        Returns:
            list: the list of data file.
        """

        if data_dirs is None:
            return None
        data_dirs = data_dirs \
            if isinstance(data_dirs, (list, tuple)) else [data_dirs]

        suffix = '' if suffix is None else suffix

        if exclude is None:
            exclude = []
        else:
            assert isinstance(exclude, (list, tuple))

        files = []
        for data_dir in data_dirs:
            for f in mmcv.scandir(data_dir, suffix=suffix):
                if f not in exclude:
                    files.append(osp.join(data_dir, f))
        files.sort()
        return files
