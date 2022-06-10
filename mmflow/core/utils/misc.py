# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from mmflow.core import FlowDataSample


def unpack_flow_data_samples(
        batch_data_samples: Sequence[FlowDataSample]
) -> Tuple[Optional[Tensor]]:
    batch_gt_flow_fw = []
    batch_gt_flow_bw = []
    batch_gt_occ_fw = []
    batch_gt_occ_bw = []
    batch_gt_valid = []
    for data_sample in batch_data_samples:
        if hasattr(data_sample, 'gt_flow_fw'):
            batch_gt_flow_fw.append(data_sample.gt_flow_fw.data[None, ...])
        if hasattr(data_sample, 'gt_flow_bw'):
            batch_gt_flow_bw.append(data_sample.gt_flow_bw.data[None, ...])
        if hasattr(data_sample, 'gt_occ_fw'):
            batch_gt_occ_fw.append(data_sample.gt_occ_fw.data[None, ...])
        if hasattr(data_sample, 'gt_occ_bw'):
            batch_gt_occ_bw.append(data_sample.gt_occ_bw.data[None, ...])
        if hasattr(data_sample, 'gt_valid'):
            batch_gt_valid.append(data_sample.gt_valid.valid)
    batch_gt_flow_fw = torch.cat(
        batch_gt_flow_fw, dim=0) if len(batch_gt_flow_fw) > 0 else None
    batch_gt_flow_bw = torch.cat(
        batch_gt_flow_bw, dim=0) if len(batch_gt_flow_bw) > 0 else None
    batch_gt_occ_fw = torch.cat(
        batch_gt_occ_fw, dim=0) if len(batch_gt_occ_fw) > 0 else None
    batch_gt_occ_bw = torch.cat(
        batch_gt_occ_bw, dim=0) if len(batch_gt_occ_bw) > 0 else None
    batch_gt_valid = torch.cat(
        batch_gt_valid, dim=0) if len(batch_gt_valid) > 0 else None
    return (batch_gt_flow_fw, batch_gt_flow_bw, batch_gt_occ_fw,
            batch_gt_occ_bw, batch_gt_valid)


def stack_batch(img1s: Sequence[Tensor],
                img2s: Sequence[Tensor]) -> Tuple[Tensor]:
    """Stack multiple tensors to form a batch and pad the images to the max
    shape use the right bottom padding mode in these images.

    Args:
        img1s (Sequence[Tensor]): The input multiple tensors for the reference
            images. Each is a CHW 3D-tensor.
        img2s (Sequence[Tensor]): The input multiple tensors for the target
            images. Each is a CHW 3D-tensor.

    Returns:
       Tuple: The 4D-tensor for reference images and target images.
    """
    assert isinstance(img1s, list), \
        f'Expected input type to be list, but got {type(img1s)}'
    assert len(set([tensor.ndim for tensor in img1s])) == 1, \
        f'Expected the dimensions of all tensors must be the same, ' \
        f'but got {[tensor.ndim for tensor in img1s]}'
    assert img1s[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
                               f'but got {img1s[0].ndim}'
    assert len(set([tensor.shape[0] for tensor in img1s])) == 1, \
        f'Expected the channels of all tensors must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in img1s]}'
    return torch.stack(img1s, dim=0), torch.stack(img2s, dim=0)
