# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Any

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class CrowdFlow(BaseDataset):
    """CrowdFlow dataset."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """load data information."""

        self.subset_dir = 'test' if self.test_mode else 'train'

        self.data_root = osp.join(self.data_root, self.subset_dir)
        self.img1_dir = osp.join(self.data_root, 'images')
        self.img2_dir = osp.join(self.data_root, 'images')
        self.flow_root = osp.join(self.data_root, 'gt_flow')

        self.img_suffix = '.png'
        self.flow_suffix = '.flo'

        self.all_scene = os.listdir(self.img1_dir)

        img1_filenames = []
        img2_filenames = []
        flow_filenames = []

        for s in self.all_scene:
            img_file_dir = osp.join(self.img1_dir, s)
            flow_file_dir = osp.join(self.flow_root, s)

            flow_filenames_ = self.get_data_filename(flow_file_dir,
                                                     self.flow_suffix)
            img_filenames_ = self.get_data_filename(img_file_dir,
                                                    self.img_suffix)
            flow_num = len(flow_filenames_)
            for i in range(flow_num):
                flow_filenames += [flow_filenames_[i]]
                img1_filenames += [img_filenames_[i]]
                img2_filenames += [img_filenames_[i + 1]]

        # img1_filenames, img2_filenames = self._revise_dir(flow_filenames)
        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, flow_filenames, 'filename_flow')
