# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from glob import glob
from typing import Callable, List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset as MME_BaseDataset

from mmflow.registry import DATASETS
from .utils import load_ann_info, load_img_info


@DATASETS.register_module()
class HD1K(MME_BaseDataset):
    """HD1K dataset.

    Args:
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

    def load_data_info(self) -> None:
        """Load data information, including file path of image1, image2 and
        optical flow."""

        self.img1_dir = osp.join(self.data_root, 'hd1k_input/image_2')
        self.img2_dir = osp.join(self.data_root, 'hd1k_input/image_2')
        self.flow_dir = osp.join(self.data_root, 'hd1k_flow_gt/flow_occ')

        img1_filenames = []
        img2_filenames = []
        flow_filenames = []

        seq_ix = 0
        while True:
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

        load_img_info(self.data_list, img1_filenames, img2_filenames)
        load_ann_info(self.data_list, flow_filenames, 'flow_fw_path')
