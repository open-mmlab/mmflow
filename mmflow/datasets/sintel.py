# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Optional, Sequence, Union

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Sintel(BaseDataset):
    """Sintel optical flow dataset.

    Args:
        pass_style (str): Pass style for Sintel dataset, and it has 2 options
            ['clean', 'final']. Default: 'clean'.
        scene (str, list, optional): Scene in Sintel dataset, if scene is None,
            it means collecting data in all of scene of Sintel dataset.
            Default: None.
    """

    def __init__(self,
                 *args,
                 pass_style: str = 'clean',
                 scene: Optional[Union[str, Sequence[str]]] = None,
                 **kwargs) -> None:

        all_pass = ['clean', 'final']
        assert pass_style in all_pass
        self.pass_style = pass_style

        self.scene = scene
        super().__init__(*args, **kwargs)

        self.dataset_name += f' {self.pass_style} subset'

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
                data_filenames_ = self.get_data_filename(
                    data_dir_, data_suffix)
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
        flow_filenames = get_filenames(self.flow_dir, self.flow_suffix)
        occ_filenames = get_filenames(self.occ_dir, self.occ_suffix)
        invalid_filenames = get_filenames(self.invalid_dir,
                                          self.invalid_suffix, 1)

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)
        self.load_ann_info(self.data_infos, flow_filenames, 'filename_flow')
        self.load_ann_info(self.data_infos, occ_filenames, 'filename_occ')
        self.load_ann_info(self.data_infos, invalid_filenames,
                           'filename_invalid')

    def _get_data_dir(self) -> None:
        """Get the paths for images and optical flow."""
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'
        self.flow_suffix = '.flo'
        self.occ_suffix = '.png'
        self.invalid_suffix = '.png'

        self.subset_dir = 'training' if self.test_mode else 'training'

        self.data_root = osp.join(self.data_root, self.subset_dir)

        img_root = osp.join(self.data_root, self.pass_style)
        flow_root = osp.join(self.data_root, 'flow')
        occ_root = osp.join(self.data_root, 'occlusions')
        invalid_root = osp.join(self.data_root, 'invalid')

        all_scene = os.listdir(img_root)
        self.scene = all_scene if self.scene is None else self.scene
        self.scene = self.scene if isinstance(self.scene,
                                              (list, tuple)) else [self.scene]
        assert set(self.scene).issubset(set(all_scene))

        self.img1_dir = [osp.join(img_root, s) for s in self.scene]
        self.img2_dir = [osp.join(img_root, s) for s in self.scene]
        self.flow_dir = [osp.join(flow_root, s) for s in self.scene]
        self.occ_dir = [osp.join(occ_root, s) for s in self.scene]
        self.invalid_dir = [osp.join(invalid_root, s) for s in self.scene]

    def pre_pipeline(self, results: Sequence[dict]) -> None:
        """Prepare results dict for pipeline.

        For Sintel, there is an additional annotation, invalid.
        """
        super().pre_pipeline(results)
        results['filename_invalid'] = results['ann_info']['filename_invalid']
