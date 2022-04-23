# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
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
        ann_file: Annotation file path. Defaults to None.
        test_mode (bool): Whether the dataset works for model testing or
            training.
    """

    def __init__(self,
                 data_root: str,
                 pipeline: Sequence[dict],
                 ann_file: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk'),
                 test_mode: bool = False) -> None:
        super().__init__()
        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.dataset_name = self.__class__.__name__
        self.file_client_args = file_client_args
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

        if ann_file is None:
            warnings.warn(message='ann_file is None, please use '
                          'tools/prepare_dataset to generate ann_file')
            self.load_data_info()
        else:
            self.load_ann_file(ann_file)

    def load_ann_file(self, ann_file):
        """_summary_

        Args:
            ann_file (_type_): _description_
        """
        ann = mmcv.load(
            ann_file,
            file_format='json',
            file_client_args=self.file_client_args)
        self.data_infos = ann['data_list']
        self.img1_dir = osp.join(self.data_root,
                                 self.data_infos[0]['img1_dir'])
        self.img2_dir = osp.join(self.data_root,
                                 self.data_infos[0]['img2_dir'])
        self.flow_dir = osp.join(self.data_root,
                                 self.data_infos[0]['flow_dir'])
        for data_info in self.data_infos:
            data_info['img_info']['filename1'] = \
                osp.join(self.img1_dir, data_info['img_info']['filename1'])
            data_info['img_info']['filename2'] = \
                osp.join(self.img2_dir, data_info['img_info']['filename2'])
            data_info['ann_info']['filename_flow'] = osp.join(
                self.data_root, data_info['ann_info']['filename_flow'])

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
