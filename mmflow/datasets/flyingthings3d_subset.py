# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
from typing import Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset as MME_BaseDataset

from mmflow.registry import DATASETS
from .utils import get_data_filename, load_ann_info, load_img_info

# these files contain nan, so exclude them.
exclude_files = dict(
    left_into_future=[
        '0004573.flo',
        '0006336.flo',
        '0016948.flo',
        '0015148.flo',
        '0006922.flo',
        '0003147.flo',
        '0003149.flo',
        '0000879.flo',
        '0006337.flo',
        '0014658.flo',
        '0015748.flo',
        '0001717.flo',
        '0000119.flo',
        '0017578.flo',
        '0004118.flo',
        '0004117.flo',
        '0004304.flo',
        '0004154.flo',
        '0011530.flo',
    ],
    right_into_future=[
        '0006336.flo',
        '0003148.flo',
        '0004117.flo',
        '0003666.flo',
    ],
    left_into_past=[
        '0000162.flo',
        '0004705.flo',
        '0006878.flo',
        '0004876.flo',
        '0004045.flo',
        '0000053.flo',
        '0005055.flo',
        '0000163.flo',
        '0000161.flo',
        '0000121.flo',
        '0000931.flo',
        '0005054.flo',
    ],
    right_into_past=[
        '0006878.flo',
        '0003147.flo',
        '0001549.flo',
        '0000053.flo',
        '0005034.flo',
        '0003148.flo',
        '0005055.flo',
        '0000161.flo',
        '0001648.flo',
        '0000160.flo',
        '0005054.flo',
    ])


@DATASETS.register_module()
class FlyingThings3DSubset(MME_BaseDataset):
    """FlyingThings3D subset dataset.

    Args:
        scene (str, optional): Scene in Flyingthings3D subset dataset,
            which has 3 options None, 'left' and 'right', and None means
            ``data_list`` will not be filter by scene filed.
            Defaults to None.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img=None, ann=None).
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Defaults to None which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.
    """
    METAINFO = dict()

    def __init__(self,
                 scene: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000) -> None:

        if scene is not None:
            assert scene in (
                'left', 'right'
            ), f'`scene` expected to be \'left\' or \'right\' but got {scene}'
        self.scene = scene

        super().__init__(ann_file, metainfo, data_root, data_prefix,
                         filter_cfg, indices, serialize_data, pipeline,
                         test_mode, lazy_init, max_refetch)

    def load_data_list(self) -> List[dict]:
        """Load ``data_list``

        ``data_list`` can be load from an annotation file named as
        ``self.ann_file`` or by parsing dataset path.

        Returns:
            list[dict]: A list of annotation.
        """
        if self.ann_file.endswith('json'):
            # load data_list with annotation file
            return super().load_data_list()
        else:
            # load data_list by path parsing
            self.load_data_info()
            return self.data_list

    def filter_data(self) -> List[dict]:
        """Filter data_list according to  ``scene``

        Returns:
            list[int]: Filtered results.
        """
        if self.scene is None:
            return self.data_list
        else:

            return [
                data_info for data_info in self.data_list
                if data_info['scene'] == self.scene
            ]

    def _get_data_dir(self):
        """Get the paths for images and optical flow."""

        self.flow_fw_dir = 'into_future'
        self.flow_bw_dir = 'into_past'
        self.flow_suffix = '.flo'
        self.occ_suffix = '.png'
        self.img1_suffix = '.png'
        self.img2_suffix = '.png'

        self.subset_dir = 'val' if self.test_mode else 'train'

        self.data_root = osp.join(self.data_root, self.subset_dir)
        img_root = osp.join(self.data_root, 'image_clean')
        flow_root = osp.join(self.data_root, 'flow')
        occ_root = osp.join(self.data_root, 'flow_occlusions')

        all_scene = os.listdir(img_root)

        self.img1_dir = [osp.join(img_root, s) for s in all_scene]
        self.img2_dir = [osp.join(img_root, s) for s in all_scene]
        self.flow_dir = [osp.join(flow_root, s) for s in all_scene]
        self.occ_dir = [osp.join(occ_root, s) for s in all_scene]

        self.flow_fw_dir = [osp.join(d, 'into_future') for d in self.flow_dir]
        self.flow_bw_dir = [osp.join(d, 'into_past') for d in self.flow_dir]
        self.occ_fw_dir = [osp.join(d, 'into_future') for d in self.occ_dir]
        self.occ_bw_dir = [osp.join(d, 'into_past') for d in self.occ_dir]

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""

        self._get_data_dir()

        img1_filenames = []
        img2_filenames = []
        flow_fw_filenames = []
        flow_bw_filenames = []
        tmp_flow_fw_filenames = []
        tmp_flow_bw_filenames = []
        occ_fw_filenames = []
        occ_bw_filenames = []
        for _flow_fw_dir, _flow_bw_dir in zip(self.flow_fw_dir,
                                              self.flow_bw_dir):
            scene = _flow_fw_dir.split(os.sep)[-2]

            exc_key_fw = scene + '_into_future'
            exc_key_bw = scene + '_into_past'

            flow_fw_filenames_ = get_data_filename(_flow_fw_dir, None,
                                                   exclude_files[exc_key_fw])
            flow_bw_filenames_ = get_data_filename(_flow_bw_dir, None,
                                                   exclude_files[exc_key_bw])
            tmp_flow_fw_filenames += flow_fw_filenames_
            tmp_flow_bw_filenames += flow_bw_filenames_

        for ii in range(len(tmp_flow_fw_filenames)):
            flo_fw = tmp_flow_fw_filenames[ii]
            img1, img2, occ_fw, occ_bw, flo_bw = self._revise_dir(flo_fw)

            if (not (flo_bw in tmp_flow_bw_filenames) or not osp.isfile(img1)
                    or not osp.isfile(img2) or not osp.isfile(occ_fw)
                    or not osp.isfile(occ_bw)):
                continue
            img1_filenames.append(img1)
            img2_filenames.append(img2)
            flow_fw_filenames.append(flo_fw)
            flow_bw_filenames.append(flo_bw)
            occ_fw_filenames.append(occ_fw)
            occ_bw_filenames.append(occ_bw)

        load_img_info(self.data_list, img1_filenames, img2_filenames)
        load_ann_info(self.data_list, flow_fw_filenames, 'flow_fw_path')
        load_ann_info(self.data_list, occ_fw_filenames, 'occ_fw_path')
        load_ann_info(self.data_list, flow_bw_filenames, 'flow_bw_path')
        load_ann_info(self.data_list, occ_bw_filenames, 'occ_bw_path')

        for i in range(len(self.data_list)):
            self.data_list[i]['scene'] = self.data_list[i]['img1_path'].split(
                osp.sep)[-2]

    def _revise_dir(self, flow_fw_filename) -> Sequence[str]:
        """Revise directory of optical flow to get the directories of image and
        occlusion mask.

        Args:
            flow_fw_filename (str):  abstract file path of optical flow from
                image1 to image2.
        Returns:
            Sequence[str]: abstract paths of image1, image2, forward
                occlusion mask, backward occlusion mask and backward optical
                flow.
        """

        idx_f = int(osp.splitext(osp.basename(flow_fw_filename))[0])

        img1_filename = flow_fw_filename.replace(
            f'{os.sep}flow{os.sep}', f'{os.sep}image_clean{os.sep}')
        img1_filename = img1_filename.replace(f'{os.sep}into_future{os.sep}',
                                              f'{os.sep}')

        img1_filename = img1_filename.replace(self.flow_suffix,
                                              self.img1_suffix)
        img2_filename = re.sub(r'\d{7}', f'{idx_f+1:07d}', img1_filename)

        flow_bw_filename = flow_fw_filename.replace(
            f'{os.sep}into_future{os.sep}', f'{os.sep}into_past{os.sep}')
        flow_bw_filename = re.sub(r'\d{7}', f'{idx_f+1:07d}', flow_bw_filename)

        occ_fw_filename = flow_fw_filename.replace(
            f'{os.sep}flow{os.sep}', f'{os.sep}flow_occlusions{os.sep}')

        occ_fw_filename = occ_fw_filename.replace(self.flow_suffix,
                                                  self.occ_suffix)
        occ_bw_filename = flow_bw_filename.replace(
            f'{os.sep}flow{os.sep}', f'{os.sep}flow_occlusions{os.sep}')

        occ_bw_filename = occ_bw_filename.replace(self.flow_suffix,
                                                  self.occ_suffix)

        return (img1_filename, img2_filename, occ_fw_filename, occ_bw_filename,
                flow_bw_filename)
