# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Sequence

from .base_dataset import BaseDataset
from .builder import DATASETS

# these files contain nan, so exclude them.
exclude_files = ['08755.pfm']


@DATASETS.register_module()
class ChairsSDHom(BaseDataset):
    """ChaorsSDHom dataset."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """load data information."""

        self.subset_dir = 'data/test' if self.test_mode else 'data/train'

        self.data_root = osp.join(self.data_root, self.subset_dir)
        self.img1_dir = osp.join(self.data_root, 't0')
        self.img2_dir = osp.join(self.data_root, 't1')
        self.flow_dir = osp.join(self.data_root, 'flow')

        self.img1_suffix = '.png'
        self.img2_suffix = '.png'
        self.flow_suffix = '.pfm'

        flow_filenames = self.get_data_filename(
            self.flow_dir, self.flow_suffix, exclude=exclude_files)
        img1_filenames, img2_filenames = self._revise_dir(flow_filenames)

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, flow_filenames, 'filename_flow')

    def _revise_dir(self,
                    flow_filenames: Sequence[str]) -> Sequence[Sequence[str]]:
        """Revise flow filename to filenames of img1 and img2.

        Args:
            flow_filenames (list): list of abstract file path of optical flow.

        Returns:
            Sequence[Sequence[str]]: list of abstract file path of image1 and
                image2.
        """
        img1_filenames = []
        img2_filenames = []
        for flow_filename in flow_filenames:
            idx = int(osp.splitext(osp.basename(flow_filename))[0])
            img1_filename = osp.join(self.img1_dir,
                                     f'{idx:05d}' + self.img1_suffix)
            img2_filename = osp.join(self.img2_dir,
                                     f'{idx:05d}' + self.img1_suffix)
            img1_filenames.append(img1_filename)
            img2_filenames.append(img2_filename)
        return img1_filenames, img2_filenames
