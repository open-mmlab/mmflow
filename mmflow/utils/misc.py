# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from torch import Tensor

from mmflow.data import FlowDataSample


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed. All workers must call
    this function, otherwise it will deadlock. This method is generally used in
    `DistributedSampler`, because the seed should be identical across all
    processes in the distributed group.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def unpack_flow_data_samples(
        batch_data_samples: Sequence[FlowDataSample]
) -> Tuple[Optional[Tensor]]:
    """Unpack data sample list.

    Args:
        batch_data_samples (Sequence[FlowDataSample]): The list of data samples


    Returns:
        Tuple[Optional[Tensor]]: Tuple of ground truth tensor.
    """
    batch_gt_flow_fw = []
    batch_gt_flow_bw = []
    batch_gt_occ_fw = []
    batch_gt_occ_bw = []
    batch_gt_valid_fw = []
    batch_gt_valid_bw = []

    for data_sample in batch_data_samples:
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
