# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
from mmcv import sparse_flow_from_bytes
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient

from mmflow.registry import TRANSFORMS
from ..utils import flow_from_bytes


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load image1 and image2 from file.

    Required Keys:

    - img1_path
    - img2_path

    Modified Keys:

    - img1
    - img2
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 file_client_args: dict = dict(backend='disk'),
                 imdecode_backend: str = 'cv2') -> None:
        super().__init__()
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = FileClient(**self.file_client_args)
        self.imdecode_backend = imdecode_backend

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Call function to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename1 = results['img1_path']
        img1_bytes = self.file_client.get(filename1)
        img1 = mmcv.imfrombytes(
            img1_bytes, flag=self.color_type, backend=self.imdecode_backend)

        filename2 = results['img2_path']
        img2_bytes = self.file_client.get(filename2)
        img2 = mmcv.imfrombytes(
            img2_bytes, flag=self.color_type, backend=self.imdecode_backend)

        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['img1'] = img1
        results['img2'] = img2
        results['img_shape'] = img1.shape[:2]
        results['ori_shape'] = img1.shape[:2]

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(BaseTransform):
    """Load optical flow from file.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of optical flow ground truth file.
            'flow_fw_path': 'a/b/c',
            'flow_bw_path': 'a/b/c',
            'occ_bw_path': 'a/b/c',
            'occ_fw_path': 'a/b/c',

        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {

            'gt_flow_fw': np.ndarray (H, W, 2)
            'gt_flow_bw': np.ndarray (H, W, 2)
            'gt_occ_fw': np.ndarray (H, W)
            'gt_occ_bw': np.ndarray (H, W)
            'gt_valid_fw': np.ndarray (H, W)
            'gt_valid_bw': np.ndarray (H, W)

        }

    Required Keys:

    - flow_fw_path
    - flow_bw_path (optional)
    - occ_fw_path (optional)
    - occ_bw_path (optional)

    Added Keys:

    - gt_flow_fw (np.float32)
    - gt_flow_bw (np.float32, optional)
    - gt_occ_fw (np.uint, optional)
    - gt_occ_bw (np.uint, optional)
    - gt_valid_fw (np.float32, optional)
    - gt_valid_bw (np.float32, optional)

    Args:
        with_occ (bool): whether to parse and load occlusion mask.
            Default to False.
        sparse (bool): whether the flow is sparse. Default to False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
            self,
            with_occ: bool = False,
            sparse: bool = False,
            file_client_args: dict = dict(backend='disk'),
    ) -> None:

        self.with_occ = with_occ
        self.sparse = sparse
        self.file_client_args = file_client_args
        self.file_client = None

    def transform(self, results: Dict) -> Dict:
        """Call function to load optical flow and occlusion mask (optional).

        Args:
            results (dict): Result dict from :obj:`mmflow.Dataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        if self.sparse:
            results = self._load_sparse_flow(results)
        else:
            results = self._load_flow(results)
        if self.with_occ:
            results = self._load_occ(results)

        results['sparse'] = self.sparse

        return results

    def _load_flow(self, results: dict) -> dict:
        """load dense optical flow function.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """

        flow_fw_filename = results.get('flow_fw_path', None)
        flow_bw_filename = results.get('flow_bw_path', None)

        if flow_fw_filename is not None:
            flow_fw_bytes = self.file_client.get(flow_fw_filename)
            flow_fw = flow_from_bytes(flow_fw_bytes, flow_fw_filename[-3:])
        else:
            flow_fw = None

        if flow_bw_filename is not None:
            flow_bw_bytes = self.file_client.get(flow_bw_filename)
            flow_bw = flow_from_bytes(flow_bw_bytes, flow_bw_filename[-3:])
        else:
            flow_bw = None
        results['gt_flow_fw'] = flow_fw
        results['gt_flow_bw'] = flow_bw

        return results

    def _load_sparse_flow(self, results: dict) -> dict:
        """load sparse optical flow function.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        flow_fw_filename = results.get('flow_fw_path', None)

        if flow_fw_filename is not None:
            flow_fw_bytes = self.file_client.get(flow_fw_filename)
            flow_fw, valid_fw = sparse_flow_from_bytes(flow_fw_bytes)
        else:
            flow_fw = None
            valid_fw = None

        results['gt_flow_fw'] = flow_fw
        results['gt_flow_bw'] = None
        # sparse flow dataset don't include backward flow
        results['gt_valid_fw'] = valid_fw
        results['gt_valid_bw'] = None

        return results

    def _load_occ(self, results: dict) -> dict:
        """load annotation function.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        occ_fw_filename = results.get('occ_fw_path', None)
        occ_bw_filename = results.get('occ_bw_path', None)

        if occ_fw_filename is not None:
            occ_fw_bytes = self.file_client.get(occ_fw_filename)
            occ_fw = (mmcv.imfrombytes(occ_fw_bytes, flag='grayscale') /
                      255).astype(np.float32)
        else:
            occ_fw = None
        if occ_bw_filename is not None:
            occ_bw_bytes = self.file_client.get(occ_bw_filename)
            occ_bw = (mmcv.imfrombytes(occ_bw_bytes, flag='grayscale') /
                      255).astype(np.float32)
        else:
            occ_bw = None
        results['gt_occ_fw'] = occ_fw
        results['gt_occ_bw'] = occ_bw

        return results

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__
        repr_str += f'(with_occ={self.with_occ},'
        repr_str += f"sparse='{self.sparse}',"
        repr_str += f"file_client_args='{self.file_client_args}')"

        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results: dict) -> dict:
        """Call function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img1 = results['img1']
        img2 = results['img2']
        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['img1_path'] = None
        results['img2_path'] = None
        results['img1'] = img1
        results['img2'] = img2
        results['img_shape'] = img1.shape[:2]
        results['ori_shape'] = img1.shape[:2]

        return results
