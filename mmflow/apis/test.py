# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmflow.datasets.utils import visualize_flow, write_flow

Module = torch.nn.Module
DataLoader = torch.utils.data.DataLoader


def single_gpu_test(
        model: Module,
        data_loader: DataLoader,
        out_dir: Optional[str] = None,
        show_dir: Optional[str] = None) -> Sequence[Dict[str, np.ndarray]]:
    """Test model with single gpus and save the results.

    Note:
        This function only can save dense predicted flow output.

    Args:
        model (nn.Module): The optical flow estimator model.
        data_loader (DataLoader): The test dataloader.
        show (bool): Whether render the flow to color map. `out_dir`
            must be define if True. Defaults to False.
        show_dir (str, optional): The path to save flow maps. Defaults to None.
        out_dir (str, optional): The path to save flow files. Defaults to None.

    Returns:
        list: The predicted results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(test_mode=True, **data)
        batch_size = len(result)
        results += result

        for _ in range(batch_size):
            prog_bar.update()
    if out_dir is not None:
        mmcv.mkdir_or_exist(out_dir)
        for i, r in enumerate(results):
            write_flow(r, osp.join(out_dir, f'flow_{i:03d}.flo'))

    if show_dir is not None:
        mmcv.mkdir_or_exist(show_dir)
        for i, r in enumerate(results):
            visualize_flow(r, osp.join(show_dir, f'flowmap_{i:03d}.png'))

    return results


def multi_gpu_test(
        model: Module,
        data_loader: DataLoader,
        tmpdir: Optional[str] = None,
        gpu_collect: bool = False) -> Sequence[Dict[str, np.ndarray]]:
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(test_mode=True, **data)
            if result[0].get('flow', None) is not None:
                result = [_['flow'] for _ in result]
            elif result[0].get('flow_fw', None) is not None:
                result = [_['flow_fw'] for _ in result]
        batch_size = len(result)
        results += result

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part: Any,
                        size: int,
                        tmpdir: Optional[str] = None) -> List[Any]:
    """Collect results with cpu from different devices when using DDP training.

    Args:
        result_part (any): Partial result from gpu.
        size (int): Length of the whole dataset.
        tmpdir (str, Optional): Temporary directory for dumping the result.
            Defaults to None.

    Returns:
        list: The whole result from all of gpus.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part: Any, size: int) -> List[Any]:
    """Collect results with gpu from different devices when using DDP training.

    Args:
        result_part (any): Partial result from gpu.
        size (int): Length of the whole dataset.

    Returns:
        list: The whole result from all of gpus.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
