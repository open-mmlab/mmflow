# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from glob import glob

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class HD1K(BaseDataset):
    """HD1K dataset."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""
        self.img1_dir = osp.join(self.data_root, 'hd1k_input/image_2')
        self.img2_dir = osp.join(self.data_root, 'hd1k_input/image_2')
        self.flow_dir = osp.join(self.data_root, 'hd1k_flow_gt/flow_occ')

        self.img1_suffix = '.png',
        self.img2_suffix = '.png',
        self.flow_suffix = '.png'

        img1_filenames = []
        img2_filenames = []
        flow_filenames = []

        seq_ix = 0
        while 1:
            flows = sorted(
                glob(osp.join(self.flow_dir, '%06d_*.png' % seq_ix)))
            images = sorted(
                glob(osp.join(self.img1_dir, '%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                flow_filenames += [flows[i]]
                img1_filenames += [images[i]]
                img2_filenames += [images[i + 1]]

            seq_ix += 1

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, flow_filenames, 'filename_flow')
