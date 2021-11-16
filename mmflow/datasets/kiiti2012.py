# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class KITTI2012(BaseDataset):
    """KITTI flow 2012 dataset."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_data_dir(self) -> None:
        """Get the paths for images and optical flow."""
        # only provide ground truth for training
        self.subset_dir = 'training'

        self.data_root = osp.join(self.data_root, self.subset_dir)
        # In KITTI 2012, data in `colored_0` is original data
        self.img1_dir = osp.join(self.data_root, 'colored_0')
        self.img2_dir = osp.join(self.data_root, 'colored_0')
        self.flow_dir = osp.join(self.data_root, 'flow_occ')

        self.img1_suffix = '_10.png',
        self.img2_suffix = '_11.png',
        self.flow_suffix = '_10.png'

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""
        self._get_data_dir()
        img1_filenames = self.get_data_filename(self.img1_dir,
                                                self.img1_suffix)
        img2_filenames = self.get_data_filename(self.img2_dir,
                                                self.img2_suffix)
        flow_filenames = self.get_data_filename(self.flow_dir,
                                                self.flow_suffix)

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, flow_filenames, 'filename_flow')
