# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Any, Sequence

from .base_dataset import BaseDataset
from .builder import DATASETS

@DATASETS.register_module()
class AutoFlow(BaseDataset):
    """AutoFlow dataset."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """load data information."""

        self.subset_dir = 'test' if self.test_mode else 'train'

        self.data_root = osp.join(self.data_root, self.subset_dir)
        self.img1_dir = self.data_root
        self.img2_dir = self.data_root
        self.flow_root = self.data_root

        self.img_suffix = '.png'
        self.flow_suffix = '.flo'

        self.all_scene = os.listdir(self.img1_dir)

        img1_filenames = []
        img2_filenames = []
        flow_filenames = []

        for s in self.all_scene:
            file_dir = os.listdir(osp.join(self.img1_dir,s))
            for i in file_dir:
                img_file_dir = osp.join(self.img1_dir, s,i)
                flow_file_dir = osp.join(self.flow_root, s, i)

                flow_filenames_ = self.get_data_filename(flow_file_dir, self.flow_suffix)
                img_filenames_ = self.get_data_filename(img_file_dir, self.img_suffix)

                flow_filenames.append(flow_filenames_[0])
                img1_filenames.append(img_filenames_[0])
                img2_filenames.append(img_filenames_[1])


        # img1_filenames, img2_filenames = self._revise_dir(flow_filenames)
        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, flow_filenames, 'filename_flow')

