# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv import sparse_flow_from_bytes
from mmcv.transforms import BaseTransform

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
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to ``dict(backend='local')``
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 backend_args: dict = dict(backend='local'),
                 imdecode_backend: str = 'cv2') -> None:
        super().__init__()
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args.copy()
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
        img1_bytes = fileio.get(filename1, self.backend_args)
        img1 = mmcv.imfrombytes(
            img1_bytes, flag=self.color_type, backend=self.imdecode_backend)

        filename2 = results['img2_path']
        img2_bytes = fileio.get(filename2, self.backend_args)
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
        backend_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='local')``.
    """

    def __init__(
            self,
            with_occ: bool = False,
            sparse: bool = False,
            backend_args: dict = dict(backend='local'),
    ) -> None:

        self.with_occ = with_occ
        self.sparse = sparse
        self.backend_args = backend_args

    def transform(self, results: Dict) -> Dict:
        """Call function to load optical flow and occlusion mask (optional).

        Args:
            results (dict): Result dict from :obj:`mmflow.Dataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """

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
            flow_fw_bytes = fileio.get(flow_fw_filename, self.backend_args)
            flow_fw = flow_from_bytes(flow_fw_bytes, flow_fw_filename[-3:])
        else:
            flow_fw = None

        if flow_bw_filename is not None:
            flow_bw_bytes = fileio.get(flow_bw_filename, self.backend_args)
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
            flow_fw_bytes = fileio.get(flow_fw_filename, self.backend_args)
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
            occ_fw_bytes = fileio.get(occ_fw_filename, self.backend_args)
            occ_fw = (mmcv.imfrombytes(occ_fw_bytes, flag='grayscale') /
                      255).astype(np.float32)
        else:
            occ_fw = None
        if occ_bw_filename is not None:
            occ_bw_bytes = fileio.get(occ_bw_filename, self.backend_args)
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
        repr_str += f"backend_args='{self.backend_args}')"

        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def transform(self, results: dict) -> dict:
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


@TRANSFORMS.register_module()
class InferencerLoader():
    """Input loader for flow inferencer."""

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromWebcam', **kwargs))

    def __call__(self, img1: Union[str, np.ndarray],
                 img2: Union[str, np.ndarray]) -> Dict:
        if isinstance(img1, str):
            inputs = dict(img1_path=img1, img2_path=img2)
        elif isinstance(img1, np.ndarray):
            inputs = dict(img1=img1, img2=img2)
        else:
            raise NotImplementedError

        if 'img1' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)
