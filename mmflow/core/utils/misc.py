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
    batch_gt_valid_fw = []
    batch_gt_valid_bw = []

    for data_sample in batch_data_samples:
        if hasattr(data_sample, 'gt_flow_fw'):
            batch_gt_flow_fw.append(data_sample.gt_flow_fw.data[None, ...])
        if hasattr(data_sample, 'gt_flow_bw'):
            batch_gt_flow_bw.append(data_sample.gt_flow_bw.data[None, ...])
        if hasattr(data_sample, 'gt_occ_fw'):
            batch_gt_occ_fw.append(data_sample.gt_occ_fw.data[None, ...])
        if hasattr(data_sample, 'gt_occ_bw'):
            batch_gt_occ_bw.append(data_sample.gt_occ_bw.data[None, ...])
        if hasattr(data_sample, 'gt_valid_fw'):
            batch_gt_valid_fw.append(data_sample.gt_valid_fw.data)
        if hasattr(data_sample, 'gt_valid_bw'):
            batch_gt_valid_bw.append(data_sample.gt_valid_bw.data)

    batch_gt_flow_fw = torch.cat(
        batch_gt_flow_fw, dim=0) if len(batch_gt_flow_fw) > 0 else None
    batch_gt_flow_bw = torch.cat(
        batch_gt_flow_bw, dim=0) if len(batch_gt_flow_bw) > 0 else None
    batch_gt_occ_fw = torch.cat(
        batch_gt_occ_fw, dim=0) if len(batch_gt_occ_fw) > 0 else None
    batch_gt_occ_bw = torch.cat(
        batch_gt_occ_bw, dim=0) if len(batch_gt_occ_bw) > 0 else None
    batch_gt_valid_fw = torch.cat(
        batch_gt_valid_fw, dim=0) if len(batch_gt_valid_fw) > 0 else None
    batch_gt_valid_bw = torch.cat(
        batch_gt_valid_bw, dim=0) if len(batch_gt_valid_bw) > 0 else None
    return (batch_gt_flow_fw, batch_gt_flow_bw, batch_gt_occ_fw,
            batch_gt_occ_bw, batch_gt_valid_fw, batch_gt_valid_bw)
