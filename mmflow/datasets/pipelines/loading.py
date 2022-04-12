# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv import sparse_flow_from_bytes

from ..builder import PIPELINES
from ..utils import flow_from_bytes


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load image1 and image2 from file.

    Required keys are "img1_info" (dict that must contain the key "filename"
    and "filename2"). Added or updated keys are "img1", "img2", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0, 1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 file_client_args: dict = dict(backend='disk'),
                 imdecode_backend: str = 'cv2') -> None:
        super().__init__()
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results: dict) -> dict:
        """Call function to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename1 = results['img_info']['filename1']
        filename2 = results['img_info']['filename2']
        if (not osp.isfile(filename1)) or (not osp.isfile(filename2)):

            raise RuntimeError(
                f'Cannot load file from {filename1} or {filename2}')

        img1_bytes = self.file_client.get(filename1)
        img2_bytes = self.file_client.get(filename2)

        img1 = mmcv.imfrombytes(
            img1_bytes, flag=self.color_type, backend=self.imdecode_backend)
        img2 = mmcv.imfrombytes(
            img2_bytes, flag=self.color_type, backend=self.imdecode_backend)

        assert img1 is not None

        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['filename1'] = filename1
        results['filename2'] = filename2
        results['ori_filename1'] = osp.split(filename1)[-1]
        results['ori_filename2'] = osp.split(filename2)[-1]

        results['img1'] = img1
        results['img2'] = img2

        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img1.shape
        results['scale_factor'] = np.array([1.0, 1.0])
        num_channels = 1 if len(img1.shape) < 3 else img1.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    """Load optical flow from file.

    Args:
        with_occ (bool): whether to parse and load occlusion mask.
            Default to False.
        sparse (bool): whether the flow is sparse. Default to False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
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

    def __call__(self, results: dict) -> dict:
        """Call function to load optical flow and occlusion mask (optional).

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if self.sparse:
            results = self._load_sparse_flow(results)
        else:
            results = self._load_flow(results)
        if self.with_occ:
            results = self._load_occ(results)
        return results

    def _load_flow(self, results: dict) -> dict:
        """load dense optical flow function.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        filenames = list(results['ann_info'].keys())
        skip_len = len('filename_')

        for filename in filenames:

            if filename.find('flow') > -1:

                filename_flow = results['ann_info'][filename]
                flow_bytes = self.file_client.get(filename_flow)
                flow = flow_from_bytes(flow_bytes, filename_flow[-3:])

                results[filename] = filename_flow
                results['ori_' + filename] = osp.split(filename_flow)[-1]
                ann_key = filename[skip_len:] + '_gt'
                results[ann_key] = flow
                results['ann_fields'].append(ann_key)

        return results

    def _load_sparse_flow(self, results: dict) -> dict:
        """load sparse optical flow function.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        filenames = list(results['ann_info'].keys())
        skip_len = len('filename_')

        for filename in filenames:

            if filename.find('flow') > -1:

                filename_flow = results['ann_info'][filename]
                flow_bytes = self.file_client.get(filename_flow)
                flow, valid = sparse_flow_from_bytes(flow_bytes)

                results[filename] = filename_flow
                results['ori_' + filename] = osp.split(filename_flow)[-1]
                ann_key = filename[skip_len:] + '_gt'
                # sparse flow dataset don't include backward flow
                results['valid'] = valid
                results[ann_key] = flow
                results['ann_fields'].append(ann_key)

        return results

    def _load_occ(self, results: dict) -> dict:
        """load annotation function.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains loaded annotation data.
        """
        filenames = list(results['ann_info'].keys())
        skip_len = len('filename_')

        for filename in filenames:

            if filename.find('occ') > -1:

                filename_occ = results['ann_info'][filename]
                occ_bytes = self.file_client.get(filename_occ)
                occ = (mmcv.imfrombytes(occ_bytes, flag='grayscale') /
                       255).astype(np.float32)

                results[filename] = filename_occ
                results['ori_' + filename] = osp.split(filename_occ)[-1]
                ann_key = filename[skip_len:] + '_gt'
                results[ann_key] = occ
                results['ann_fields'].append(ann_key)

        return results


@PIPELINES.register_module()
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

        results['filename1'] = None
        results['ori_filename1'] = None
        results['filename2'] = None
        results['ori_filename2'] = None
        results['img1'] = img1
        results['img2'] = img2
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        results['img_fields'] = ['img1', 'img2']
        # Set initial values for default meta_keys
        results['pad_shape'] = img1.shape
        results['scale_factor'] = np.array([1.0, 1.0])

        return results
