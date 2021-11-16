# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class FlyingChairs(BaseDataset):
    """FlyingChairs dataset.

    Args:
        split_file (str): File name of train-validation split file for
            FlyingChairs.
    """

    def __init__(self, *args, split_file: str, **kwargs) -> None:

        self.split = np.loadtxt(split_file, dtype=np.int32).tolist()
        super().__init__(*args, **kwargs)

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

        img1_filenames = self.get_data_filename(self.img1_dir,
                                                self.img1_suffix)
        img2_filenames = self.get_data_filename(self.img2_dir,
                                                self.img2_suffix)
        flow_filenames = self.get_data_filename(self.flow_dir,
                                                self.flow_suffix)

        assert len(img1_filenames) == len(img2_filenames) == len(
            flow_filenames)

        self.load_img_info(img1_filenames, img2_filenames)
        self.load_ann_info(flow_filenames, 'filename_flow')

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
                    img_info=dict(
                        filename1=img1_filename[i],
                        filename2=img2_filename[i]),
                    ann_info=dict())
                self.data_infos.append(data_info)

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
                self.data_infos[
                    i - num_tests]['ann_info'][filename_key] = filename[i]
            else:
                num_tests += 1
