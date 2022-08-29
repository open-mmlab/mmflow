# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from typing import Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmengine.utils import is_str

from mmflow.registry import TRANSFORMS
from mmflow.structures import FlowDataSample


def to_tensor(
    data: Union[np.ndarray, torch.Tensor, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class PackFlowInputs(BaseTransform):
    """Pack the inputs data for the optical flow model.

     The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img1_path``: path to the first image file

        - ``img2_path``: path to the first image file

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be collected in
        ``data[img_metas]``. Default: ``('img1_path', 'img2_path',
        'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')``

    Args:
        BaseTransform (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_keys = ('gt_flow_fw', 'gt_flow_bw', 'gt_occ_fw', 'gt_occ_bw',
                 'gt_valid_fw', 'gt_valid_bw')

    def __init__(
        self,
        meta_keys=('img1_path', 'img2_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad', 'pad_shape', 'flip',
                   'flip_direction')
    ) -> None:

        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:
            - 'img1' (obj:`torch.Tensor`): The first image data for models.
            - 'img2' (obj:`torch.Tensor`): The second image data for models.
            - 'data_sample' (obj:`FlowDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        data_sample = FlowDataSample()
        if 'img1' in results:
            img1 = results['img1']
            img1 = np.expand_dims(img1, -1) if len(img1.shape) < 3 else img1
            img1 = to_tensor(np.ascontiguousarray(img1.transpose(2, 0, 1)))
        if 'img2' in results:
            img2 = results['img2']
            img2 = np.expand_dims(img2, -1) if len(img2.shape) < 3 else img2
            img2 = to_tensor(np.ascontiguousarray(img2.transpose(2, 0, 1)))
        # inputs shape 2,3,H,W for image1 and image2
        packed_results['inputs'] = torch.stack((img1, img2), dim=0)

        for key in self.data_keys:
            if results.get(key, None) is not None:
                ann_data = results[key]
                if len(ann_data.shape) < 3:
                    ann_data = to_tensor(ann_data[None, ...])
                else:
                    ann_data = to_tensor(
                        np.ascontiguousarray(ann_data.transpose(2, 0, 1)))
                data = PixelData(**dict(data=ann_data))
                data_sample.set_data({key: data})

        img_meta = dict()
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
