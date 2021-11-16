# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from glob import glob
from typing import Sequence, Union

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class FlyingThings3D(BaseDataset):
    """FlyingThings3D subset dataset.

    Args:
        direction (str): Direction of flow, has 4 options 'forward',
            'backward', 'bidirection' and ['forward', 'backward'].
            Default: ['forward', 'backward'].
        scene (list, str): Scene in Flyingthings3D dataset, default: 'left'.
            This default value is for RAFT, as FlyingThings3D is so large
            and not often used, and only RAFT use the 'left' data in it.
        pass_style (str): Pass style for FlyingThing3D dataset, and it has 2
            options ['clean', 'final']. Default: 'clean'.
    """

    def __init__(self,
                 *args,
                 direction: Union[str,
                                  Sequence[str]] = ['forward', 'backward'],
                 scene: Union[str, Sequence[str]] = 'left',
                 pass_style: str = 'clean',
                 **kwargs) -> None:

        assert direction in ['forward', 'backward', 'bidirection'
                             ] or direction == ['forward', 'backward']
        self.direction = direction

        scene = scene if isinstance(scene, (list, tuple)) else [scene]
        assert set(scene).issubset(['left', 'right'])
        self.scene = scene

        assert pass_style in ['clean', 'final']
        self.pass_style = pass_style

        super().__init__(*args, **kwargs)

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""
        self._get_data_dir()

        img1_filenames = []
        img2_filenames = []
        flow_fw_filenames = []
        flow_bw_filenames = []
        flow_filenames = []

        for idir, fw_dir, bw_dir in zip(self.img1_dir, self.flow_fw_dir,
                                        self.flow_bw_dir):

            img_filenames_ = self.get_data_filename(idir, self.img1_suffix)
            img_filenames_.sort()
            flow_fw_filenames_ = self.get_data_filename(
                fw_dir, self.flow_suffix)
            flow_fw_filenames.sort()
            flow_bw_filenames_ = self.get_data_filename(
                bw_dir, self.flow_suffix)
            flow_bw_filenames.sort()

            if self.direction == 'forward':
                img1_filenames += img_filenames_[:-1]
                img2_filenames += img_filenames_[1:]
                flow_filenames += flow_fw_filenames_[:-1]
            elif self.direction == 'backward':
                img1_filenames += img_filenames_[1:]
                img2_filenames += img_filenames_[:-1]
                flow_filenames += flow_bw_filenames_[1:]
            elif self.direction == 'bidirection':
                img1_filenames += img_filenames_[:-1]
                img2_filenames += img_filenames_[1:]
                flow_fw_filenames += flow_fw_filenames_[:-1]
                flow_bw_filenames += flow_bw_filenames_[1:]
            else:
                img1_filenames += img_filenames_[:-1]
                img2_filenames += img_filenames_[1:]
                flow_filenames += flow_fw_filenames_[:-1]
                img1_filenames += img_filenames_[1:]
                img2_filenames += img_filenames_[:-1]
                flow_filenames += flow_bw_filenames_[1:]

        self.load_img_info(self.data_infos, img1_filenames, img2_filenames)

        if self.direction == 'bidirection':
            self.load_ann_info(self.data_infos, flow_fw_filenames,
                               'filename_flow_fw')
            self.load_ann_info(self.data_infos, flow_bw_filenames,
                               'filename_flow_bw')

        else:
            self.load_ann_info(self.data_infos, flow_filenames,
                               'filename_flow')

    def _get_data_dir(self) -> None:
        """Get the paths for images and optical flow."""

        self.flow_fw_dir = 'into_future'
        self.flow_bw_dir = 'into_past'
        self.flow_suffix = '.pfm'
        self.occ_suffix = '.png'
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'

        self.subset_dir = 'TEST' if self.test_mode else 'TRAIN'

        pass_dir = 'frames_' + self.pass_style + 'pass'
        imgs_dirs = []
        flow_fw_dirs = []
        flow_bw_dirs = []

        for scene in self.scene:
            imgs_dirs_ = glob(
                osp.join(self.data_root, pass_dir, self.subset_dir + '/*/*'))
            imgs_dirs_ = [osp.join(f, scene) for f in imgs_dirs_]
            imgs_dirs += imgs_dirs_

            flow_fw_dirs_ = glob(
                osp.join(self.data_root, 'optical_flow',
                         self.subset_dir + '/*/*/' + self.flow_fw_dir))

            flow_fw_dirs_ = [osp.join(f, scene) for f in flow_fw_dirs_]
            flow_fw_dirs += flow_fw_dirs_

            flow_bw_dirs_ = glob(
                osp.join(self.data_root, 'optical_flow',
                         self.subset_dir + '/*/*/' + self.flow_bw_dir))
            flow_bw_dirs_ = [osp.join(f, scene) for f in flow_bw_dirs_]
            flow_bw_dirs += flow_bw_dirs_

        self.img1_dir = imgs_dirs
        self.img2_dir = imgs_dirs
        self.flow_fw_dir = flow_fw_dirs
        self.flow_bw_dir = flow_bw_dirs
