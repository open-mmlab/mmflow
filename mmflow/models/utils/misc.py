# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from mmflow.structures import FlowDataSample


def unpack_flow_data_samples(
        data_samples: Sequence[FlowDataSample]) -> Tuple[Optional[Tensor]]:
    """Unpack data sample list.

    Args:
        data_samples (Sequence[FlowDataSample]): The list of data samples


    Returns:
        Tuple[Optional[Tensor]]: Tuple of ground truth tensor.
    """
    batch_gt_flow_fw = []
    batch_gt_flow_bw = []
    batch_gt_occ_fw = []
    batch_gt_occ_bw = []
    batch_gt_valid_fw = []
    batch_gt_valid_bw = []

    for data_sample in data_samples:
        if hasattr(data_sample, 'gt_flow_fw'):
            batch_gt_flow_fw.append(data_sample.gt_flow_fw.data)
        if hasattr(data_sample, 'gt_flow_bw'):
            batch_gt_flow_bw.append(data_sample.gt_flow_bw.data)
        if hasattr(data_sample, 'gt_occ_fw'):
            batch_gt_occ_fw.append(data_sample.gt_occ_fw.data)
        if hasattr(data_sample, 'gt_occ_bw'):
            batch_gt_occ_bw.append(data_sample.gt_occ_bw.data)
        if hasattr(data_sample, 'gt_valid_fw'):
            batch_gt_valid_bw.append(data_sample.gt_valid_fw.data)
        if hasattr(data_sample, 'gt_valid_bw'):
            batch_gt_valid_bw.append(data_sample.gt_valid_bw.data)

    batch_gt_flow_fw = torch.stack(
        batch_gt_flow_fw) if len(batch_gt_flow_fw) > 0 else None
    batch_gt_flow_bw = torch.stack(
        batch_gt_flow_bw) if len(batch_gt_flow_bw) > 0 else None
    batch_gt_occ_fw = torch.stack(
        batch_gt_occ_fw) if len(batch_gt_occ_fw) > 0 else None
    batch_gt_occ_bw = torch.stack(
        batch_gt_occ_bw) if len(batch_gt_occ_bw) > 0 else None
    batch_gt_valid_fw = torch.cat(
        batch_gt_valid_fw, dim=0) if len(batch_gt_valid_fw) > 0 else None
    batch_gt_valid_bw = torch.cat(
        batch_gt_valid_bw, dim=0) if len(batch_gt_valid_bw) > 0 else None
    return (batch_gt_flow_fw, batch_gt_flow_bw, batch_gt_occ_fw,
            batch_gt_occ_bw, batch_gt_valid_fw, batch_gt_valid_bw)
